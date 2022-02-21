#!/usr/bin/env python
"""Training on a single process."""
import torch
#import deepspeed

from onmt.inputters.inputter import IterOnDevice
from onmt.model_builder import build_model
from onmt.utils.optimizers import Optimizer
from onmt.utils.misc import set_random_seed
from onmt.trainer import build_trainer
from onmt.models import build_model_saver
from onmt.utils.logging import init_logger, logger
from onmt.utils.parse import ArgumentParser

from onmt.utils.distributed import all_reduce_tensors_init
from onmt.inputters.dynamic_iterator import build_dynamic_dataset_iter
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from collections import OrderedDict


def configure_process(opt, device_id):
    logger.info("logger set device {} ".format(device_id))
    if device_id >= 0:
        torch.cuda.set_device(device_id)
    set_random_seed(opt.seed, device_id >= 0)


def _get_model_opts(opt, checkpoint=None):
    """Get `model_opt` to build model, may load from `checkpoint` if any."""
    if checkpoint is not None:
        model_opt = ArgumentParser.ckpt_model_opts(checkpoint["opt"])
        ArgumentParser.update_model_opts(model_opt)
        ArgumentParser.validate_model_opts(model_opt)
        if (opt.tensorboard_log_dir == model_opt.tensorboard_log_dir and
                hasattr(model_opt, 'tensorboard_log_dir_dated')):
            # ensure tensorboard output is written in the directory
            # of previous checkpoints
            opt.tensorboard_log_dir_dated = model_opt.tensorboard_log_dir_dated
        # Override checkpoint's update_embeddings as it defaults to false
        model_opt.update_vocab = opt.update_vocab
    else:
        model_opt = opt
    return model_opt


def _build_valid_iter(opt, fields, transforms_cls):
    """Build iterator used for validation."""
    valid_iter = build_dynamic_dataset_iter(
        fields, transforms_cls, opt, is_train=False)
    return valid_iter


def _build_train_iter(opt, fields, transforms_cls, stride=1, offset=0, nodeID=-1, gpuID=-1):
    """Build training iterator."""
    train_iter_map = build_dynamic_dataset_iter(
        fields, transforms_cls, opt, is_train=True,
        stride=stride, offset=offset, nodeID=nodeID, gpuID=gpuID)
    return train_iter_map


def create_group_distributed(model_opt, numGPUS, nodeRank, device_id):
    node_gpu_langsFile = open(model_opt.node_gpu_langs, 'rt')
    dictLangEncoder_toIDXset = OrderedDict()
    dictLangDecoder_toIDXset = OrderedDict()
    langsEnc = OrderedDict()
    langsDec = OrderedDict()
#    numLangPairONthisdevice = 0
    myuniqueidx = numGPUS * nodeRank + device_id
    for line in node_gpu_langsFile:
        nodeIDXgpuIdx_langpair = line.strip().split(" ")
        nodeIDX = int(nodeIDXgpuIdx_langpair[0])
        gpuIDX = int(nodeIDXgpuIdx_langpair[1])
        langpair = nodeIDXgpuIdx_langpair[2].split("-")

        if nodeIDX == nodeRank and gpuIDX == device_id:
#            numLangPairONthisdevice+=1
            if langpair[0] in langsEnc:
                countIDXlp = langsEnc[langpair[0]]+1
            else:
                countIDXlp = 1
            langsEnc[langpair[0]] = countIDXlp

            if langpair[1] in langsDec:
                countIDXlp = langsDec[langpair[1]]+1
            else:
                countIDXlp = 1
            langsDec[langpair[1]] = countIDXlp


#            langsEnc.add(langpair[0])
#            langsDec.add(langpair[1])
        dist_rank = numGPUS * nodeIDX + gpuIDX

        if not langpair[0] in dictLangEncoder_toIDXset:
            idxset = set()
        else:
            idxset = dictLangEncoder_toIDXset[langpair[0]]
        idxset.add(dist_rank)
        dictLangEncoder_toIDXset[langpair[0]] = idxset

        if not langpair[1] in dictLangDecoder_toIDXset:
            idxsetD = set()
        else:
            idxsetD = dictLangDecoder_toIDXset[langpair[1]]
        idxsetD.add(dist_rank)
        dictLangDecoder_toIDXset[langpair[1]] = idxsetD

    node_gpu_langsFile.close()
    dictLangEncoder_toGroup = OrderedDict()
    dictLangDecoder_toGroup = OrderedDict()

    for k in dictLangEncoder_toIDXset:
        if len(list(dictLangEncoder_toIDXset[k])) ==1:
            continue
        logger.info("%s encoderGroup %s unique-IDX %s",str(k),str(list(sorted(dictLangEncoder_toIDXset[k]))), myuniqueidx)
        dictLangEncoder_toGroup[k] = dist.new_group(list(sorted(dictLangEncoder_toIDXset[k])))
    for k in dictLangDecoder_toIDXset:
        if len(list(dictLangDecoder_toIDXset[k])) ==1:
            continue
        logger.info("%s decoderGroup %s unique-IDX %s", str(k),str(list(sorted(dictLangDecoder_toIDXset[k]))), myuniqueidx)
        dictLangDecoder_toGroup[k] = dist.new_group(list(sorted(dictLangDecoder_toIDXset[k])))


    return dictLangEncoder_toGroup, dictLangDecoder_toGroup, langsEnc, langsDec


def init_distributed(model, dictLangEncoder_toGroup, dictLangDecoder_toGroup, langsEnc, langsDec, world_size):
    for langsource in dictLangEncoder_toGroup.keys():
        if langsource in langsEnc:
            groupE = dictLangEncoder_toGroup[langsource]
            weightsEnc = [p.data for p in model.encoder["encoder" + str(langsource)].parameters()]
            we = [*weightsEnc]
            all_reduce_tensors_init(we, float(langsEnc[str(langsource)]), groupE)

    for langtarget in dictLangDecoder_toGroup.keys():
        if langtarget in langsDec:
            groupD = dictLangDecoder_toGroup[langtarget]
            weightsDec = [p.data for p in model.decoder["decoder" + str(langtarget)].parameters()]
            wd = [*weightsDec]
            all_reduce_tensors_init(wd, float(langsDec[str(langtarget)]), groupD)
            logger.info(langtarget)
            if str(langtarget) == 'cs':
                for p in model.decoder["decoder" + str(langtarget)].parameters():
                    logger.info(p[0:10])
                    break
 
            weightsGen = [p.data for p in model.generator["generator" + str(langtarget)].parameters()]
            wg = [*weightsGen]
            all_reduce_tensors_init(wg, float(langsDec[str(langtarget)]), groupD)

    weightsAtt = [p.data for p in model.attention_bridge.parameters()]
    wa = [*weightsAtt]
    all_reduce_tensors_init(wa, float(world_size))


def main(opt, fields_dict, device_id,
         batch_queue=None, semaphore=None, nodeRank=None, dictLangEncoder_toGroup=None, dictLangDecoder_toGroup=None): #fields, transforms_cls, checkpoint, 
    """Start training on `device_id`."""
    # NOTE: It's important that ``opt`` has been validated and updated
    # at this point.
    unique_device_id = device_id
    numGPUSperNode = len(opt.gpu_ranks)
    device_id = device_id % numGPUSperNode 

    world_size = int(opt.world_size)

    configure_process(opt, device_id)
    init_logger(opt.log_file)

    gpu_rankT = torch.distributed.get_rank()
    logger.info("RANK GPU FROM TORCH %s", str(gpu_rankT))

    #checkpoint, fields, transforms_cls = _init_train(opt)
    transforms_cls = None
    checkpoint = None
    model_opt = _get_model_opts(opt, checkpoint=checkpoint)

    dictLangEncoder_toGroup, dictLangDecoder_toGroup, langsEnc, langsDec = create_group_distributed(model_opt, numGPUSperNode, nodeRank, device_id)

    #logger.info(dictLangEncoder_toGroup)
    #logger.info(dictLangDecoder_toGroup)
    logger.info("BUILDING MODEL")
    # Build model.
    model, generators_md = build_model(model_opt, opt, fields_dict, checkpoint, nodeRank, device_id) #build_model(model_opt, opt, fields, checkpoint, nodeRank, device_id)
    logger.info("INIT MODEL")
    init_distributed(model, dictLangEncoder_toGroup, dictLangDecoder_toGroup, langsEnc, langsDec, world_size)
    model.count_parameters(log=logger.info)
    logger.info("LANGS DEVICE")
    logger.info(langsEnc)
    logger.info(langsDec)

#    model, optim, _, _ = deepspeed.initialize(config="/scratch/project_2005099/members/raganato/ds_config.json", model=model, model_parameters=model.parameters())
    logger.info("MODEL DDDP")
#    model = DDP(modelz, device_ids=[device_id], output_device=device_id, find_unused_parameters=True)
    """
    print("MODEL DDP")
    print(model)
    print("MODEL DDP2")
    for name, param in model.named_parameters():
          print(name)
    print("THEN")
    """
    # Build optimizer.
    logger.info("BUILD OPTMIZER")
    #optims = {}
    #for le in langsEnc.keys():
    #    menc = model.encoder["encoder" + str(le)]
    #    optimE = Optimizer.from_opt(menc, opt, checkpoint=checkpoint)
    #    optimE.zero_grad()
    #    optims["ENC_"+str(le)] = optimE
    #for ld in langsDec.keys():
    #    mdec = model.decoder["decoder" + str(ld)]
    #    optimD = Optimizer.from_opt(mdec, opt, checkpoint=checkpoint)
    #    optimD.zero_grad()
    #    optims["DEC_"+str(ld)] = optimD
    #    mgen = model.generator["generator" + str(ld)]
    #    optimG = Optimizer.from_opt(mgen, opt, checkpoint=checkpoint)
    #    optimG.zero_grad()
    #    optims["GEN_"+str(ld)] = optimG

    #mATT = model.attention_bridge
    #optimA = Optimizer.from_opt(mATT, opt, checkpoint=checkpoint)
   # optimA.zero_grad()
    #optims["attention_bridge"] = optimA

    optim = Optimizer.from_opt(model, opt, checkpoint=checkpoint, langsEnc=langsEnc, langsDec=langsDec)

    #print("TRIANINGI STEPS")
    #print(optim.training_step)
    # Build model saver
    model_saver = build_model_saver(model_opt, opt, model, fields_dict, optim, unique_device_id)
    logger.info("BUILD TRAINER")
    trainer = build_trainer(
        opt, device_id, model, fields_dict, optim, model_saver=model_saver, generators_md=generators_md,  dictLangEncoder_toGroup=dictLangEncoder_toGroup, dictLangDecoder_toGroup=dictLangDecoder_toGroup)
    logger.info("DONE BUILD TRAINER")
    if batch_queue is None:
        _train_iter_map = _build_train_iter(opt, fields_dict, transforms_cls, stride=1, offset=0, nodeID=0, gpuID=0)
        train_iter = IterOnDevice(_train_iter_map, device_id)
    else:
        assert semaphore is not None, \
            "Using batch_queue requires semaphore as well"

        def _train_iter():
            while True:
                (batch, langPairname) = batch_queue.get()
#                print("TRAIN SINGLE ITER")
#                print(batch)
#                print(langPairname) #wmt_en-tr
                semaphore.release()
                # Move batch to specified device
                IterOnDevice.batch_to_device(batch, device_id)
                yield batch, langPairname


        train_iter = _train_iter()
    logger.info("VALID ITER")
    valid_iter = _build_valid_iter(opt, fields_dict, transforms_cls)
    if valid_iter is not None:
        valid_iter = IterOnDevice(valid_iter, device_id)

    if len(opt.gpu_ranks):
        logger.info('Starting training on GPU: %s' % opt.gpu_ranks)
    else:
        logger.info('Starting training on CPU, could be very slow')
    train_steps = opt.train_steps
    if opt.single_pass and train_steps > 0:
        logger.warning("Option single_pass is enabled, ignoring train_steps.")
        train_steps = 0
    logger.info("TRAIN GO")
    trainer.train(
        train_iter,
        train_steps,
        save_checkpoint_steps=opt.save_checkpoint_steps,
        valid_iter=valid_iter,
        valid_steps=opt.valid_steps, unique_devide_id=unique_device_id)

    if trainer.report_manager.tensorboard_writer is not None:
        trainer.report_manager.tensorboard_writer.close()
