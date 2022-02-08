"""
    This is the loadable seq2seq trainer library that is
    in charge of training details, loss compute, and statistics.
    See train.py for a use case of this library.

    Note: To make this a general library, we implement *only*
          mechanism things here(i.e. what to do), and leave the strategy
          things to users(i.e. how to do it). Also see train.py(one of the
          users of this library) for the strategy things we do.
"""

import torch
import traceback

import onmt.utils
from onmt.utils.logging import logger
import torch.nn as nn
from collections import OrderedDict
from operator import add
import torch.distributed

def build_trainer(opt, device_id, model, Fields_dict, optim, model_saver=None, generators_md=None,
                  dictLangEncoder_toGroup=None, dictLangDecoder_toGroup=None):
    """
    Simplify `Trainer` creation based on user `opt`s*

    Args:
        opt (:obj:`Namespace`): user options (usually from argument parsing)
        model (:obj:`onmt.models.NMTModel`): the model to train
        fields (dict): dict of fields
        optim (:obj:`onmt.utils.Optimizer`): optimizer used during training
        data_type (str): string describing the type of data
            e.g. "text"
        model_saver(:obj:`onmt.models.ModelSaverBase`): the utility object
            used to save the model
    """

    #    tgt_field = dict(fields)["tgt"].base_field
    train_loss_md = nn.ModuleDict()
    valid_loss_md = nn.ModuleDict()
    logger.info("BUILD TRAINER")
    #    for name, generator in generators_md.items():
    # nameLang = str(name).replace("generator", "")
    targetLangs = set()
    for src_tgt_lang in Fields_dict.keys():
        tgt_field = dict(Fields_dict[src_tgt_lang])["tgt"].base_field
        nameLang = src_tgt_lang.split("-")[1]
        if nameLang in targetLangs:
            continue
        targetLangs.add(nameLang)
        logger.info("BEFORE")
        logger.info(nameLang)
        if not "generator" + nameLang in generators_md:
            continue
        logger.info("AFTER")
        logger.info(nameLang)
        generator = generators_md["generator" + nameLang]
        train_loss_md.add_module('trainloss{0}'.format(nameLang),
                                 onmt.utils.loss.build_loss_compute(model, tgt_field, opt, train=True,
                                                                    generator=generator))  # [name] = onmt.utils.loss.build_loss_compute(model, tgt_field, opt, train=True, generator=generator)
        valid_loss_md.add_module('valloss{0}'.format(nameLang),
                                 onmt.utils.loss.build_loss_compute(model, tgt_field, opt, train=False,
                                                                    generator=generator))  # [name] = onmt.utils.loss.build_loss_compute(model, tgt_field, opt, train=False,  generator=generator)
    #    train_loss = onmt.utils.loss.build_loss_compute(model, tgt_field, opt, generator=generator)
    #    valid_loss = onmt.utils.loss.build_loss_compute(
    #        model, tgt_field, opt, train=False,  generator=generator)

    trunc_size = opt.truncated_decoder  # Badly named...
    shard_size = opt.max_generator_batches if opt.model_dtype == 'fp32' else 0
    norm_method = opt.normalization
    accum_count = opt.accum_count
    accum_steps = opt.accum_steps
    n_gpu = opt.world_size
    average_decay = opt.average_decay
    average_every = opt.average_every
    dropout = opt.dropout
    dropout_steps = opt.dropout_steps
    if device_id >= 0:
        gpu_rank = opt.gpu_ranks[device_id]
    else:
        gpu_rank = -1
        n_gpu = 0
    gpu_verbose_level = opt.gpu_verbose_level

    earlystopper = onmt.utils.EarlyStopping(
        opt.early_stopping, scorers=onmt.utils.scorers_from_opts(opt)) \
        if opt.early_stopping > 0 else None

    report_manager = onmt.utils.build_report_manager(opt, gpu_rank)
    trainer = onmt.Trainer(model, train_loss_md, valid_loss_md, optim, trunc_size,
                           shard_size, norm_method,
                           accum_count, accum_steps,
                           n_gpu, gpu_rank,
                           gpu_verbose_level, report_manager,
                           with_align=True if opt.lambda_align > 0 else False,
                           model_saver=model_saver,  # if gpu_rank <= 0 else None,
                           average_decay=average_decay,
                           average_every=average_every,
                           model_dtype=opt.model_dtype,
                           earlystopper=earlystopper,
                           dropout=dropout,
                           dropout_steps=dropout_steps, dictLangEncoder_toGroup=dictLangEncoder_toGroup,
                           dictLangDecoder_toGroup=dictLangDecoder_toGroup)
    return trainer


class Trainer(object):
    """
    Class that controls the training process.

    Args:
            model(:py:class:`onmt.models.model.NMTModel`): translation model
                to train
            train_loss(:obj:`onmt.utils.loss.LossComputeBase`):
               training loss computation
            valid_loss(:obj:`onmt.utils.loss.LossComputeBase`):
               training loss computation
            optim(:obj:`onmt.utils.optimizers.Optimizer`):
               the optimizer responsible for update
            trunc_size(int): length of truncated back propagation through time
            shard_size(int): compute loss in shards of this size for efficiency
            data_type(string): type of the source input: [text]
            norm_method(string): normalization methods: [sents|tokens]
            accum_count(list): accumulate gradients this many times.
            accum_steps(list): steps for accum gradients changes.
            report_manager(:obj:`onmt.utils.ReportMgrBase`):
                the object that creates reports, or None
            model_saver(:obj:`onmt.models.ModelSaverBase`): the saver is
                used to save a checkpoint.
                Thus nothing will be saved if this parameter is None
    """

    def __init__(self, model, train_loss_md, valid_loss_md, optim,
                 trunc_size=0, shard_size=32,
                 norm_method="sents", accum_count=[1],
                 accum_steps=[0],
                 n_gpu=1, gpu_rank=1, gpu_verbose_level=0,
                 report_manager=None, with_align=False, model_saver=None,
                 average_decay=0, average_every=1, model_dtype='fp32',
                 earlystopper=None, dropout=[0.3], dropout_steps=[0], dictLangEncoder_toGroup=None,
                 dictLangDecoder_toGroup=None):
        # Basic attributes.
        self.model = model
        self.train_loss_md = train_loss_md
        self.valid_loss_md = valid_loss_md
        self.optim = optim
        self.trunc_size = trunc_size
        self.shard_size = shard_size
        self.norm_method = norm_method
        self.accum_count_l = accum_count
        self.accum_count = accum_count[0]
        self.accum_steps = accum_steps
        self.n_gpu = n_gpu
        self.gpu_rank = gpu_rank
        self.gpu_verbose_level = gpu_verbose_level
        self.report_manager = report_manager
        self.with_align = with_align
        self.model_saver = model_saver
        self.average_decay = average_decay
        self.moving_average = None
        self.average_every = average_every
        self.model_dtype = model_dtype
        self.earlystopper = earlystopper
        self.dropout = dropout
        self.dropout_steps = dropout_steps

        self.dictLangEncoder_toGroup = dictLangEncoder_toGroup
        self.dictLangDecoder_toGroup = dictLangDecoder_toGroup
        #TODO
        self.numLangPairONthisdevice = 4 #numLangPairONthisdevice

        #self.training_step_all = 1
        self.unique_devide_id =None
        for i in range(len(self.accum_count_l)):
            assert self.accum_count_l[i] > 0
            if self.accum_count_l[i] > 1:
                assert self.trunc_size == 0, \
                    """To enable accumulated gradients,
                       you must disable target sequence truncating."""

        # Set model in training mode.
        self.model.train()

    def _accum_count(self, step):
        # print("AACOMCONT")
        # print(step)
        for i in range(len(self.accum_steps)):
            if step > self.accum_steps[i]:
                _accum = self.accum_count_l[i]
        return _accum

    def _maybe_update_dropout(self, step):
        for i in range(len(self.dropout_steps)):
            if step > 1 and step == self.dropout_steps[i] + 1:
                self.model.update_dropout(self.dropout[i])
                logger.info("Updated dropout to %f from step %d"
                            % (self.dropout[i], step))

    def _accum_batches(self, iterator):
        batches = []
        normalization = 0
        #self.accum_count = self._accum_count(self.training_step_all)
        self.accum_count = self._accum_count(self.optim.training_step)
        # print("ACCUM BATCH")
        for batch, langPairname in iterator:
            # print(langPairname) #wmt_en-tr
            SrcTgt = str(langPairname).split("_")[1].split("-")
            sourceLang = SrcTgt[0]
            targetLang = SrcTgt[1]
            # print(sourceLang)
            # print(targetLang)
            batches.append(batch)
            if self.norm_method == "tokens":
                num_tokens = batch.tgt[1:, :, 0].ne(
                    self.train_loss_md["trainloss" + targetLang].padding_idx).sum()
                normalization += num_tokens.item()
            else:
                normalization += batch.batch_size
            if len(batches) == self.accum_count:
                yield batches, normalization, sourceLang, targetLang
                self.accum_count = self._accum_count(self.optim.training_step)
                #self.accum_count = self._accum_count(self.training_step_all)
                batches = []
                normalization = 0
        if batches:
            yield batches, normalization, sourceLang, targetLang

    def _update_average(self, step):
        if self.moving_average is None:
            copy_params = [params.detach().float()
                           for params in self.model.parameters()]
            self.moving_average = copy_params
        else:
            average_decay = max(self.average_decay,
                                1 - (step + 1) / (step + 10))
            for (i, avg), cpt in zip(enumerate(self.moving_average),
                                     self.model.parameters()):
                self.moving_average[i] = \
                    (1 - average_decay) * avg + \
                    cpt.detach().float() * average_decay

    def train(self,
              train_iter,
              train_steps,
              save_checkpoint_steps=5000,
              valid_iter=None,
              valid_steps=10000, unique_devide_id=None):
        """
        The main training loop by iterating over `train_iter` and possibly
        running validation on `valid_iter`.

        Args:
            train_iter: A generator that returns the next training batch.
            train_steps: Run training for this many iterations.
            save_checkpoint_steps: Save a checkpoint every this many
              iterations.
            valid_iter: A generator that returns the next validation batch.
            valid_steps: Run evaluation every this many iterations.

        Returns:
            The gathered statistics.
        """
        if valid_iter is None:
            logger.info('Start training loop without validation...')
        else:
            logger.info('Start training loop and validate every %d steps...',
                        valid_steps)

        total_stats = onmt.utils.Statistics()
        report_stats = onmt.utils.Statistics()
        self._start_report_manager(start_time=total_stats.start_time)
        self.unique_devide_id =unique_devide_id
        self.optim.zero_grad()
        trainEnum = enumerate(self._accum_batches(train_iter))
        #print("ENURMEATE")
        #print(trainEnum)
        #print(next(trainEnum))
        # WHILE TRUE
        #self.numLangPairONthisdevice*=4
        
        #if self.unique_devide_id ==0 or self.unique_devide_id ==2:
        #    dataw = [p.data for p in self.model.decoder["decoderet"].parameters()]
        #    logger.info("devide_id: %s - %s", str(self.unique_devide_id), str(dataw[2][:10]))


        while True:

            step =  self.optim.training_step #self.training_step_all # self.optim.training_step
            self._maybe_update_dropout(step)

            #grads_enc_communication = OrderedDict()
            #grads_dec_communication = OrderedDict()
            #grads_att_communication = OrderedDict()
            #grads_gen_communication = OrderedDict()
            langsEnc = set()
            langsDec = set()

            for j in range(self.numLangPairONthisdevice):
                i, (batches, normalization, sourceLang, targetLang) = next(trainEnum)
                # step = self.optim.training_step
                # self._maybe_update_dropout(step)
                #logger.info("%s - %s -RANK: %s", str(sourceLang),str(targetLang),str(unique_devide_id))

                if self.n_gpu > 1:
                    normalization = sum(onmt.utils.distributed
                                        .all_gather_list
                                        (normalization))

                grads_enc, grads_dec, grads_att, grads_gen = self._gradient_accumulation_overLangPair(
                    batches, normalization, total_stats,
                    report_stats, sourceLang, targetLang)

                #self.update_dict_grads(grads_enc, grads_enc_communication, sourceLang)
                #self.update_dict_grads(grads_dec, grads_dec_communication, targetLang)
                #self.update_dict_grads(grads_gen, grads_gen_communication, targetLang)
                #self.update_dict_grads(grads_att, grads_att_communication, "attention_bridge")
                langsEnc.add(sourceLang)
                langsDec.add(targetLang)


            sourceLang = str(langsEnc)
            targetLang = str(langsDec)
            #logger.info(sourceLang + " - "+ targetLang)
            #logger.info("ENC")
            
            #logger.info("START Unique devide id %s",self.unique_devide_id)
            
            for langsource in self.dictLangEncoder_toGroup.keys():
                if langsource in langsEnc:
                    #logger.info("%s EncoderGrads - Unique devide id %s",langsource, self.unique_devide_id)
                    groupE = self.dictLangEncoder_toGroup[langsource]
                    #logger.info(groupE)
                    #gradse = grads_enc_communication[langsource]
                    gradse = [p.grad.data for p in self.model.encoder["encoder" + str(langsource)].parameters() if
                                p.requires_grad and p.grad is not None]
                    #logger.info(gradse)
                    onmt.utils.distributed.all_reduce_and_rescale_tensors(gradse, float(1), groupE)
                    #idxgra=0
                    #for p in self.model.encoder["encoder"+str(langsource)].parameters():
                    #    p.grad.data = gradse[idxgra]
                    #    idxgra+=1


            
            #onmt.utils.distributed.all_reduce_and_rescale_tensors(gradse, float(1))#, group)
            #print("DEC and GEN")
            for langtarget in self.dictLangDecoder_toGroup.keys():
                if langtarget in langsDec:
#                    logger.info("%s DecoderGrads - Unique devide id %s",langtarget, self.unique_devide_id)
                    groupD = self.dictLangDecoder_toGroup[langtarget]
                    #logger.info(groupD)
                    #gradsd = grads_dec_communication[langtarget]
                    gradsd = [p.grad.data for p in self.model.decoder["decoder" + str(langtarget)].parameters() if
                                p.requires_grad and p.grad is not None]

                    onmt.utils.distributed.all_reduce_and_rescale_tensors(gradsd, float(1), groupD)
                    #idxgra=0
                    #for p in self.model.decoder["decoder"+str(langtarget)].parameters():
                    #    p.grad.data = gradsd[idxgra]
                    #    idxgra+=1
                    #if str(langtarget) == 'cs':
                    #    for p in self.model.decoder["decoder" + str(langtarget)].parameters():
                    #        logger.info("Traindevice "+str(self.unique_devide_id)+" "+str(p[5:10]))
                    #        break

#                    if (self.unique_devide_id ==0 or self.unique_devide_id ==2) and langtarget =="et" :
#                        logger.info("%s DecoderGrads - Unique devide id %s - %s - %s",langtarget, self.unique_devide_id, str(step), str(gradsd[2][:10]))
#                        idxgra=0
#                        for p in self.model.decoder["decoder"+str(langtarget)].parameters():
#                            p.grad.data = gradsd[idxgra]
#                            idxgra+=1
                #onmt.utils.distributed.all_reduce_and_rescale_tensors(gradsd, float(1))#, group)
                #for t in gradsd:
                #    torch.distributed.all_reduce(t)
                    #logger.info("%s GeneratorGrads - Unique devide id %s",langtarget, self.unique_devide_id)
                    #gradsg = grads_gen_communication[langtarget]
                    gradsg = [p.grad.data for p in self.model.generator["generator" + str(langtarget)].parameters() if
                                p.requires_grad and p.grad is not None]
                    onmt.utils.distributed.all_reduce_and_rescale_tensors(gradsg, float(1), groupD)
                    #idxgra=0
                    #for p in self.model.generator["generator"+str(langtarget)].parameters():
                    #    p.grad.data = gradsg[idxgra]
                    #    idxgra+=1

                #onmt.utils.distributed.all_reduce_and_rescale_tensors(gradsg, float(1))#, group)
                #for t in gradsg:
                #    torch.distributed.all_reduce(t)
                        

            #langsource = "en"
            #gradse = grads_enc_communication[langsource]
            #logger.info("EN case ENC")
            #onmt.utils.distributed.all_reduce_and_rescale_tensors(gradse, float(1))

            #langtarget = "en"
            #gradsd = grads_dec_communication[langtarget]
            #logger.info("EN case DEC")
            #onmt.utils.distributed.all_reduce_and_rescale_tensors(gradsd, float(1))

            #gradsg = grads_gen_communication[langtarget]
            #logger.info("EN case GEN")
            #onmt.utils.distributed.all_reduce_and_rescale_tensors(gradsg, float(1))


            #print("ATT")
            #logger.info("%s AttentionGrads - Unique devide id %s",sourceLang, self.unique_devide_id)
            #gradsa = grads_att_communication["attention_bridge"]
            gradsa = [p.grad.data for p in self.model.attention_bridge.parameters() if
                            p.requires_grad and p.grad is not None]
            onmt.utils.distributed.all_reduce_and_rescale_tensors(gradsa, float(1))
            #idxgra=0
            #for p in self.model.attention_bridge.parameters():
            #    p.grad.data = gradsa[idxgra]
            #    idxgra+=1

            #for t in gradsa:
            #    torch.distributed.all_reduce(t)
            #logger.info("done comm grads - Unique devide id %s",self.unique_devide_id)

            #TODO
            #for le in langsEnc:
            #    optimEnc  = self.optim["ENC_"+str(le)]
            #    optimEnc.step()
            #    optimEnc.zero_grad()
            #for ld in langsDec:
            #    optimDec  = self.optim["DEC_"+str(ld)]
            #    optimDec.step()
            #    optimDec.zero_grad()
            #    optimGen  = self.optim["GEN_"+str(ld)]
            #    optimGen.step()
            #    optimGen.zero_grad()

            #optimATT = self.optim["attention_bridge"]
            ##optimATT.step()
            #optimATT.zero_grad()

            #if self.unique_devide_id ==0 or self.unique_devide_id ==2:
            #    dataw = [p.data for p in self.model.decoder["decoderet"].parameters()]
            #    datagw = [p.grad.data for p in self.model.decoder["decoderet"].parameters()]
            #    logger.info("devide_id: %s - BEFORE %s - %s - %s", str(self.unique_devide_id), str(step), str(dataw[2][:10]), str(datagw[2][:10]))

            self.optim.step()#langsEnc, langsDec)
            #self.training_step_all+=1
            self.optim.zero_grad()
            #learning_rate_to_show = optimATT.learning_rate()

            #if self.unique_devide_id ==0 or self.unique_devide_id ==2:
            #    dataw = [p.data for p in self.model.decoder["decoderet"].parameters()]
            #    datagw = [p.grad.data for p in self.model.decoder["decoderet"].parameters()]
            #    logger.info("devide_id: %s - AFTER %s - %s - %s", str(self.unique_devide_id), str(step), str(dataw[2][:10]), str(datagw[2][:10]))


           # logger.info("done optim - Unique devide id %s",self.unique_devide_id)

            if self.average_decay > 0 and i % self.average_every == 0:
                self._update_average(step)


            report_stats = self._maybe_report_training(
                step, train_steps,
                self.optim.learning_rate(), #learning_rate_to_show, #self.optim.learning_rate(),
                report_stats, (sourceLang, targetLang))

            if valid_iter is not None and step % valid_steps == 0:
                if self.gpu_verbose_level > 0:
                    logger.info('GpuRank %d: validate step %d'
                                % (self.gpu_rank, step))
                valid_stats = self.validate(
                    valid_iter, moving_average=self.moving_average, sourceLang=sourceLang, targetLang=targetLang)
                if self.gpu_verbose_level > 0:
                    logger.info('GpuRank %d: gather valid stat \
                                    step %d' % (self.gpu_rank, step))
                valid_stats = self._maybe_gather_stats(valid_stats)
                if self.gpu_verbose_level > 0:
                    logger.info('GpuRank %d: report stat step %d'
                                % (self.gpu_rank, step))
                self._report_step(self.optim.learning_rate(), #learning_rate_to_show, #self.optim.learning_rate(),
                                  step, valid_stats=valid_stats)
                # Run patience mechanism
                if self.earlystopper is not None:
                    self.earlystopper(valid_stats, step)
                    # If the patience has reached the limit, stop training
                    if self.earlystopper.has_stopped():
                        break

            if (self.model_saver is not None
                    and (save_checkpoint_steps != 0
                         and step % save_checkpoint_steps == 0)):
                self.model_saver.save(step, moving_average=self.moving_average)

            if train_steps > 0 and step >= train_steps:
                break

        if self.model_saver is not None:
            self.model_saver.save(step, moving_average=self.moving_average)
        return total_stats

    
    def update_dict_grads(self, gradsIN, grads_dict, lang):
        if gradsIN == None:
            return
        if not lang in grads_dict:
            grads_dict[lang] = gradsIN
        else:
            grads = grads_dict[lang]
            my_list = [grads, gradsIN]
            result = torch.stack(my_list).sum(dim=0)        
            gradsTot = result #[*map(add, grads, gradsIN)] #list( )
            #gradsTot = grads + gradsIN
            grads_dict[lang] = gradsTot
    
    
    def validate(self, valid_iter, moving_average=None, sourceLang=None, targetLang=None):
        """ Validate model.
            valid_iter: validate data iterator
        Returns:
            :obj:`nmt.Statistics`: validation loss statistics
        """
        valid_model = self.model
        if moving_average:
            # swap model params w/ moving average
            # (and keep the original parameters)
            model_params_data = []
            for avg, param in zip(self.moving_average,
                                  valid_model.parameters()):
                model_params_data.append(param.data)
                param.data = avg.data.half() if self.optim._fp16 == "legacy" \
                    else avg.data
    
        # Set model in validating mode.
        valid_model.eval()
    
        with torch.no_grad():
            stats = onmt.utils.Statistics()
    
            for batch in valid_iter:
                src, src_lengths = batch.src if isinstance(batch.src, tuple) \
                    else (batch.src, None)
                tgt = batch.tgt
    
                with torch.cuda.amp.autocast(enabled=self.optim.amp):
                    # F-prop through the model.
                    outputs, attns = valid_model(src, tgt, src_lengths,
                                                 with_align=self.with_align, src_task=sourceLang, tgt_task=targetLang)
    
                    # Compute loss.
                    _, batch_stats = self.valid_loss_md["valloss" + targetLang](batch, outputs, attns)
    
                # Update statistics.
                stats.update(batch_stats)
        if moving_average:
            for param_data, param in zip(model_params_data,
                                         self.model.parameters()):
                param.data = param_data
    
        # Set model back to training mode.
        valid_model.train()
    
        return stats
    
    
    def _gradient_accumulation_overLangPair(self, true_batches, normalization, total_stats,
                                            report_stats, sourceLang, targetLang):
        grads_enc = None  # _communication = OrderedDict()
        grads_dec = None  # _communication = OrderedDict()
        grads_att = None  # _communication = OrderedDict()
        grads_gen = None  # _communication = OrderedDict()
    
        for k, batch in enumerate(true_batches):
    
            target_size = batch.tgt.size(0)
            # Truncated BPTT: reminder not compatible with accum > 1
            if self.trunc_size:
                trunc_size = self.trunc_size
            else:
                trunc_size = target_size
    
            src, src_lengths = batch.src if isinstance(batch.src, tuple) \
                else (batch.src, None)
            if src_lengths is not None:
                report_stats.n_src_words += src_lengths.sum().item()
    
            tgt_outer = batch.tgt
    
            bptt = False
            for j in range(0, target_size - 1, trunc_size):
                # 1. Create truncated target.
                tgt = tgt_outer[j: j + trunc_size]
                #TODO: AMP == TRUE If fp16 
                #logger.info("ENABLED asd %s", self.optim["attention_bridge"].amp)
                with torch.cuda.amp.autocast(enabled=self.optim.amp): #self.optim.amp):
                    # print("BATCH DEVICE")
                    # print(str(src.device)+ " - "+ str(tgt.device))
                    outputs, attns = self.model(
                        src, tgt, src_lengths, bptt=bptt,
                        with_align=self.with_align, src_task=sourceLang, tgt_task=targetLang)
                    bptt = True
    
                    # 3. Compute loss.
                    loss, batch_stats = self.train_loss_md["trainloss" + targetLang](
                        batch,
                        outputs,
                        attns,
                        normalization=normalization,
                        shard_size=self.shard_size,
                        trunc_start=j,
                        trunc_size=trunc_size)
                    #logger.info(loss)
    
                try:
                    if loss is not None:
                        #loss.backward()               
                        self.optim.backward(loss)
                        #optimGen  = self.optim["GEN_"+str(targetLang)]
                        #optimGen.backward(loss)
                        #optimDec  = self.optim["DEC_"+str(targetLang)]
                        #optimDec.backward(loss)
                        #optimATT = self.optim["attention_bridge"]
                        #optimATT.backward(loss)
                        #optimEnc  = self.optim["ENC_"+str(sourceLang)]
                        #optimEnc.backward(loss)
    
                    total_stats.update(batch_stats)
                    report_stats.update(batch_stats)
    
                except Exception:
                    traceback.print_exc()
                    logger.info("At step %d, we removed a batch - accum %d",
                                self.training_step_all, k)
                """
                if sourceLang in self.dictLangEncoder_toGroup.keys():    
                    gradsEnc = [p.grad.data for p in self.model.encoder["encoder" + str(sourceLang)].parameters() if
                                p.requires_grad and p.grad is not None]
                    gradse = gradsEnc #[*gradsEnc]
                    if grads_enc == None:
                        grads_enc = gradse
                    else:
                        my_list = [grads_enc, gradse]
                        result = torch.stack(my_list).sum(dim=0)                        
                        grads_enc = result #[*map(add, grads_enc, gradse)] # list() 

                if targetLang in self.dictLangDecoder_toGroup.keys():    
                    gradsDec = [p.grad.data for p in self.model.decoder["decoder" + str(targetLang)].parameters() if
                                p.requires_grad and p.grad is not None]
                    gradsd = gradsDec #[*gradsDec]
                    if grads_dec == None:
                        grads_dec = gradsd
                    else:
                        my_list = [grads_dec, gradsd]
                        result = torch.stack(my_list).sum(dim=0)                        
                        grads_dec = result # torch.add(grads_dec, gradsd) #[*map(add, grads_dec, gradsd)] # list() 

                    gradsGen = [p.grad.data for p in self.model.generator["generator" + str(targetLang)].parameters() if
                                p.requires_grad and p.grad is not None]
                    gradsg = gradsGen # [*gradsGen]
                    if grads_gen == None:
                        grads_gen = gradsg
                    else:
                        my_list = [grads_gen, gradsg]
                        result = torch.stack(my_list).sum(dim=0)                        
                        grads_gen = result #torch.add(grads_gen, gradsg) #[*map(add, grads_gen, gradsg)]# list() 
    
                gradsAtt = [p.grad.data for p in self.model.attention_bridge.parameters() if
                            p.requires_grad and p.grad is not None]
                gradsa = gradsAtt #[*gradsAtt]
                if grads_att == None:
                    grads_att = gradsa
                else:
                    my_list = [grads_att, gradsa]
                    result = torch.stack(my_list).sum(dim=0)                        
                    grads_att = result #torch.add(grads_att, gradsa)# [*map(add, grads_att, gradsa)] #list() 
                """     
    

    
                #                self.update_dict_grads(gradse, grads_enc_communication, sourceLang)
                #                self.update_dict_grads(gradsd, grads_dec_communication, targetLang)
                #                self.update_dict_grads(gradsg, grads_gen_communication, targetLang)
                #                self.update_dict_grads(gradsa, grads_att_communication, "attention_bridge")
    
                # If truncated, don't backprop fully.
                if self.model.decoder["decoder" + str(targetLang)].state is not None:
                    self.model.decoder["decoder" + str(targetLang)].detach_state()
    
        return grads_enc, grads_dec, grads_att, grads_gen
    
    
    def _gradient_accumulation(self, true_batches, normalization, total_stats,
                               report_stats, sourceLang, targetLang):
        if self.accum_count > 1:
            self.optim.zero_grad()
        # print("GRADIENT ACC")
        # print(str(sourceLang)+" - "+str(targetLang))
        # deviceENC = next(self.model.encoder["encoder"+str(sourceLang)].parameters()).device
        # print(deviceENC)
        # print(next(self.model.decoder["decoder"+str(targetLang)].parameters()).device)
        # self.model.decoder["decoder"+str(sourceLang)].to(deviceENC)
        # print(next(self.model.decoder["decoder"+str(targetLang)].parameters()).device)
    
        for k, batch in enumerate(true_batches):
    
            target_size = batch.tgt.size(0)
            # Truncated BPTT: reminder not compatible with accum > 1
            if self.trunc_size:
                trunc_size = self.trunc_size
            else:
                trunc_size = target_size
    
            src, src_lengths = batch.src if isinstance(batch.src, tuple) \
                else (batch.src, None)
            if src_lengths is not None:
                report_stats.n_src_words += src_lengths.sum().item()
    
            tgt_outer = batch.tgt
    
            bptt = False
            for j in range(0, target_size - 1, trunc_size):
                # 1. Create truncated target.
                tgt = tgt_outer[j: j + trunc_size]
    
                # 2. F-prop all but generator.
                if self.accum_count == 1:
                    self.optim.zero_grad()
    
                with torch.cuda.amp.autocast(enabled=self.optim.amp):
                    # print("BATCH DEVICE")
                    # print(str(src.device)+ " - "+ str(tgt.device))
                    outputs, attns = self.model(
                        src, tgt, src_lengths, bptt=bptt,
                        with_align=self.with_align, src_task=sourceLang, tgt_task=targetLang)
                    bptt = True
    
                    # 3. Compute loss.
                    loss, batch_stats = self.train_loss_md["trainloss" + targetLang](
                        batch,
                        outputs,
                        attns,
                        normalization=normalization,
                        shard_size=self.shard_size,
                        trunc_start=j,
                        trunc_size=trunc_size)
    
                try:
                    if loss is not None:
                        self.optim.backward(loss)
    
                    total_stats.update(batch_stats)
                    report_stats.update(batch_stats)
    
                except Exception:
                    traceback.print_exc()
                    logger.info("At step %d, we removed a batch - accum %d",
                                self.optim.training_step, k) #self.training_step_all, 
    
                # 4. Update the parameters and statistics.
                if self.accum_count == 1:
                    # self.optim.step()
                    # Multi GPU gradient gather
                    if self.n_gpu > 1:
                        gradsEnc = [p.grad.data for p in self.model.encoder["encoder" + str(sourceLang)].parameters()
                                    if p.requires_grad
                                    and p.grad is not None]
                        gradse = [*gradsEnc]
                        group = self.dictLangEncoder_toGroup[sourceLang]
                        # torch.distributed.all_reduce(gradse, group=group)
                        onmt.utils.distributed.all_reduce_and_rescale_tensors(gradse, float(1), group)
    
                        gradsATT = self.model.attention_bridge.parameters()
                        gradsa = [*gradsATT]
                        # torch.distributed.all_reduce(gradsa, group=group)
                        onmt.utils.distributed.all_reduce_and_rescale_tensors(gradsa, float(1))
    
                        gradsDec = [p.grad.data for p in self.model.decoder["decoder" + str(targetLang)].parameters()
                                    if p.requires_grad
                                    and p.grad is not None]
                        gradsd = [*gradsDec]
                        group = self.dictLangDecoder_toGroup[targetLang]
                        # torch.distributed.all_reduce(gradsd, group=group)
                        onmt.utils.distributed.all_reduce_and_rescale_tensors(gradsd, float(1), group)
    
                        gradsGEN = [p.grad.data for p in self.model.generator["generator" + str(targetLang)].parameters()
                                    if p.requires_grad
                                    and p.grad is not None]
                        gradsg = [*gradsGEN]
                        group = self.dictLangDecoder_toGroup[targetLang]
                        # torch.distributed.all_reduce(gradsg, group=group)
                        onmt.utils.distributed.all_reduce_and_rescale_tensors(gradsg, float(1), group)
    
                        """
                        grads = [p.grad.data for p in self.model.parameters()
                                 if p.requires_grad
                                 and p.grad is not None]
    
                        gradsEnc = [p.grad.data for p in self.model.encoder["encoder"+str(sourceLang)].parameters()
                                 if p.requires_grad
                                 and p.grad is not None]
                        gradsATT = [p.grad.data for p in self.model.attention_bridge.parameters()
                                 if p.requires_grad
                                 and p.grad is not None]
                        gradsDec = [p.grad.data for p in self.model.decoder["decoder"+str(targetLang)].parameters()
                                 if p.requires_grad
                                 and p.grad is not None]
                        gradsGen = [p.grad.data for p in self.model.generator["generator"+str(targetLang)].parameters()
                                 if p.requires_grad
                                 and p.grad is not None]
    
                        grads = [*gradsEnc, *gradsATT, *gradsDec, *gradsGen] 
    
                        onmt.utils.distributed.all_reduce_and_rescale_tensors(
                            grads, float(1))
                        """
    
                    self.optim.step()
                    # self.training_step_all+=1
    
                # If truncated, don't backprop fully.
                # TO CHECK
                # if dec_state is not None:
                #    dec_state.detach()
                if self.model.decoder["decoder" + str(targetLang)].state is not None:
                    self.model.decoder["decoder" + str(targetLang)].detach_state()
    
        # in case of multi step gradient accumulation,
        # update only after accum batches
        # TODO
        if self.accum_count > 1:
            if self.n_gpu > 1:
                grads = [p.grad.data for p in self.model.parameters()
                         if p.requires_grad
                         and p.grad is not None]
                onmt.utils.distributed.all_reduce_and_rescale_tensors(
                    grads, float(1))
            self.optim.step()
    
            # self.training_step_all +=1
    
    
    def _start_report_manager(self, start_time=None):
        """
        Simple function to start report manager (if any)
        """
        if self.report_manager is not None:
            if start_time is None:
                self.report_manager.start()
            else:
                self.report_manager.start_time = start_time
    
    
    def _maybe_gather_stats(self, stat):
        """
        Gather statistics in multi-processes cases
    
        Args:
            stat(:obj:onmt.utils.Statistics): a Statistics object to gather
                or None (it returns None in this case)
    
        Returns:
            stat: the updated (or unchanged) stat object
        """
        if stat is not None and self.n_gpu > 1:
            return onmt.utils.Statistics.all_gather_stats(stat)
        return stat
    
    
    def _maybe_report_training(self, step, num_steps, learning_rate,
                               report_stats, src_tgt):
        """
        Simple function to report training stats (if report_manager is set)
        see `onmt.utils.ReportManagerBase.report_training` for doc
        """
        if self.report_manager is not None:
            return self.report_manager.report_training(
                step,
                num_steps,
                learning_rate,
                None if self.earlystopper is None
                else self.earlystopper.current_tolerance,
                report_stats, src_tgt,
                multigpu=self.n_gpu > 1)
    
    
    def _report_step(self, learning_rate, step, train_stats=None,
                     valid_stats=None):
        """
        Simple function to report stats (if report_manager is set)
        see `onmt.utils.ReportManagerBase.report_step` for doc
        """
        if self.report_manager is not None:
            return self.report_manager.report_step(
                learning_rate,
                None if self.earlystopper is None
                else self.earlystopper.current_tolerance,
                step, train_stats=train_stats,
                valid_stats=valid_stats)
