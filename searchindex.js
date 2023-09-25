Search.setIndex({docnames:["CONTRIBUTING","attention_bridges","config_config","examples/Translation","index","install","main","mammoth","mammoth.inputters","mammoth.modules","mammoth.translate.translation_server","mammoth.translation","options/build_vocab","options/server","options/train","options/translate","prepare_data","quickstart","ref"],envversion:{"sphinx.domains.c":1,"sphinx.domains.changeset":1,"sphinx.domains.citation":1,"sphinx.domains.cpp":1,"sphinx.domains.index":1,"sphinx.domains.javascript":1,"sphinx.domains.math":2,"sphinx.domains.python":1,"sphinx.domains.rst":1,"sphinx.domains.std":1,"sphinx.ext.viewcode":1,sphinx:56},filenames:["CONTRIBUTING.md","attention_bridges.md","config_config.md","examples/Translation.md","index.rst","install.md","main.md","mammoth.rst","mammoth.inputters.rst","mammoth.modules.rst","mammoth.translate.translation_server.rst","mammoth.translation.rst","options/build_vocab.rst","options/server.rst","options/train.rst","options/translate.rst","prepare_data.md","quickstart.md","ref.rst"],objects:{"mammoth.Trainer":{train:[7,1,1,""],validate:[7,1,1,""]},"mammoth.models":{NMTModel:[7,0,1,""]},"mammoth.models.NMTModel":{count_parameters:[7,1,1,""],forward:[7,1,1,""]},"mammoth.modules":{AverageAttention:[9,0,1,""],Embeddings:[9,0,1,""],MultiHeadedAttention:[9,0,1,""],PositionalEncoding:[9,0,1,""]},"mammoth.modules.AverageAttention":{cumulative_average:[9,1,1,""],cumulative_average_mask:[9,1,1,""],forward:[9,1,1,""]},"mammoth.modules.Embeddings":{emb_luts:[9,1,1,""],forward:[9,1,1,""],load_pretrained_vectors:[9,1,1,""],word_lut:[9,1,1,""]},"mammoth.modules.MultiHeadedAttention":{forward:[9,1,1,""],training:[9,2,1,""],update_dropout:[9,1,1,""]},"mammoth.modules.PositionalEncoding":{forward:[9,1,1,""]},"mammoth.modules.position_ffn":{PositionwiseFeedForward:[9,0,1,""]},"mammoth.modules.position_ffn.PositionwiseFeedForward":{forward:[9,1,1,""]},"mammoth.translate":{BeamSearch:[11,0,1,""],DecodeStrategy:[11,0,1,""],GNMTGlobalScorer:[11,0,1,""],GreedySearch:[11,0,1,""],Translation:[11,0,1,""],TranslationBuilder:[11,0,1,""],Translator:[11,0,1,""]},"mammoth.translate.BeamSearch":{initialize:[11,1,1,""]},"mammoth.translate.DecodeStrategy":{advance:[11,1,1,""],block_ngram_repeats:[11,1,1,""],initialize:[11,1,1,""],maybe_update_forbidden_tokens:[11,1,1,""],maybe_update_target_prefix:[11,1,1,""],target_prefixing:[11,1,1,""],update_finished:[11,1,1,""]},"mammoth.translate.GreedySearch":{advance:[11,1,1,""],initialize:[11,1,1,""],update_finished:[11,1,1,""]},"mammoth.translate.Translation":{log:[11,1,1,""]},"mammoth.translate.Translator":{translate_batch:[11,1,1,""]},"mammoth.translate.greedy_search":{sample_with_temperature:[11,3,1,""]},"mammoth.translate.penalties":{PenaltyBuilder:[11,0,1,""]},"mammoth.translate.penalties.PenaltyBuilder":{coverage_none:[11,1,1,""],coverage_summary:[11,1,1,""],coverage_wu:[11,1,1,""],length_average:[11,1,1,""],length_none:[11,1,1,""],length_wu:[11,1,1,""]},"mammoth.translate.translation_server":{ServerModel:[10,0,1,""],ServerModelError:[10,4,1,""],Timer:[10,0,1,""],TranslationServer:[10,0,1,""]},"mammoth.translate.translation_server.ServerModel":{build_tokenizer:[10,1,1,""],detokenize:[10,1,1,""],do_timeout:[10,1,1,""],maybe_convert_align:[10,1,1,""],maybe_detokenize:[10,1,1,""],maybe_detokenize_with_align:[10,1,1,""],maybe_postprocess:[10,1,1,""],maybe_preprocess:[10,1,1,""],maybe_tokenize:[10,1,1,""],parse_opt:[10,1,1,""],postprocess:[10,1,1,""],preprocess:[10,1,1,""],rebuild_seg_packages:[10,1,1,""],to_gpu:[10,1,1,""],tokenize:[10,1,1,""],tokenizer_marker:[10,1,1,""]},"mammoth.translate.translation_server.TranslationServer":{clone_model:[10,1,1,""],list_models:[10,1,1,""],load_model:[10,1,1,""],preload_model:[10,1,1,""],run:[10,1,1,""],start:[10,1,1,""],unload_model:[10,1,1,""]},"mammoth.utils":{Optimizer:[7,0,1,""],Statistics:[7,0,1,""]},"mammoth.utils.Optimizer":{amp:[7,1,1,""],backward:[7,1,1,""],from_opt:[7,1,1,""],learning_rate:[7,1,1,""],step:[7,1,1,""],training_step:[7,1,1,""],zero_grad:[7,1,1,""]},"mammoth.utils.Statistics":{accuracy:[7,1,1,""],all_gather_stats:[7,1,1,""],all_gather_stats_list:[7,1,1,""],elapsed_time:[7,1,1,""],log_tensorboard:[7,1,1,""],output:[7,1,1,""],ppl:[7,1,1,""],update:[7,1,1,""],xent:[7,1,1,""]},"mammoth.utils.loss":{LossComputeBase:[7,0,1,""]},mammoth:{Trainer:[7,0,1,""]}},objnames:{"0":["py","class","Python class"],"1":["py","method","Python method"],"2":["py","attribute","Python attribute"],"3":["py","function","Python function"],"4":["py","exception","Python exception"]},objtypes:{"0":"py:class","1":"py:method","2":"py:attribute","3":"py:function","4":"py:exception"},terms:{"25g":5,"boolean":[7,11],"break":16,"class":[0,4,7,9,10],"default":[10,12,13,14,15,16],"export":5,"final":[1,3,11],"float":[2,9,11],"function":[0,1,2,7,9,10,11,14],"import":0,"int":[7,9,10,11],"long":0,"new":[0,1,3],"public":5,"return":[0,7,9,10,11],"static":[7,14],"true":[2,3,7,11,14,15,16],"try":[0,5],"while":2,And:0,EOS:11,For:[0,2,11,14,17],IDs:11,IFS:16,LPs:2,Not:0,One:2,The:[1,3,7,10,11,14,15],Then:[0,3],There:[1,2],These:[1,2,9,11],Use:[2,14,15],Used:11,Will:2,__init__:10,_compute_loss:7,aan:14,aan_useffn:[9,14],ab_fixed_length:14,ab_lay:14,ab_layer_norm:14,abbrevi:0,abil:9,about:0,abov:[0,11],abs:[1,14,15,18],acceler:[9,18],accept:[0,2,11],access:[1,2,5],accord:2,account:[2,5],accross:7,accum:7,accum_count:[3,7,14],accum_step:[3,7,14],accumul:[7,14],accuraci:[7,11],achiev:2,achin:18,acl:[6,18],aclweb:14,action:[9,11,14],activ:[1,5,9,14],activation_fn:9,activationfunct:9,actual:11,adadelta:14,adafactor:14,adagrad:14,adagrad_accumulator_init:14,adam:[3,14],adam_beta1:14,adam_beta2:[3,14],adamoptim:14,adamw:14,adapt:[4,6],adapter_nam:2,add:[0,3,9],added:2,adding:0,addit:[0,9,12,14,15],addition:9,address:11,adjust:2,adopt:14,advanc:[11,14],advic:0,after:[0,1,11,14],again:0,aidan:18,alexand:6,algorithm:18,align:[4,7,10,11,15],align_debug:15,alignment_head:14,alignment_lay:14,aliv:11,alive_attn:11,alive_seq:11,all:[0,2,7,9,11,12,14,15,18],all_gather_stat:7,all_gather_stats_list:7,all_preprocess:10,allennlp:0,alloc:2,allow:[0,1,2,14],almost:[11,14],alon:0,along:1,alpha:[1,11,15],alphabet:2,alreadi:[12,14,15],also:[0,2,5,7,9,14],alwai:[0,2],amp:[7,14],ani:[0,2,11,12,14],anoth:[0,1,7],antholog:14,apex:14,apex_opt_level:14,api:[0,4],api_doc:14,appear:2,append:[5,16],appli:[1,2,11,12,14,15],applic:15,appropri:11,approxim:14,architectur:[1,4],arg:[0,10],argmax:15,argpars:10,argument:[0,4],arxiv:[0,1,14,15,18],ashish:18,assig:2,assign:[2,15],assing:2,assum:[9,11],att_typ:1,attend:1,attent:[0,4,7,11,15,18],attention_bridg:7,attention_dropout:[3,14],attentionbridgenorm:1,attn:[11,15],attn_debug:[11,15],attn_typ:9,attr:10,attribut:11,augment:18,author:6,autodoc:0,autogener:14,avail:[7,10,14,15],available_model:13,averag:[9,14,15,18],average_decai:[3,7,14],average_everi:[7,14],average_output:9,averageattent:9,avg:15,avg_raw_prob:15,avoid:[0,2],aws:5,axi:11,back:7,backend:14,backward:7,bahdanau:14,ban_unk_token:[11,15],barri:18,bart:[12,14,15],base:[0,1,2,3,5,6,7,9,10,11,12,14,15],baselin:14,basemodel:7,basenam:[3,16],bash:5,batch:[1,3,7,9,11,14,15],batch_siz:[3,9,11,14,15],batch_size_multipl:[3,14],batch_typ:[3,14,15],beam:[4,11],beam_search:11,beam_siz:[3,11,15],beamsearch:11,beamsearchbas:11,becaus:[2,15],becom:2,been:[9,11,12,14,15],befor:[0,3,10,11,14,15],begin:[7,11],below:0,ben:2,bengali:2,best:[11,15],beta1:14,beta2:14,beta:[11,15],better:[0,12,14,15],between:[1,12,14,15,18],beyond:7,biao:18,bib:0,bibtex:0,bibtext:0,bidir_edg:14,bidirect:14,bin:[5,14],binari:[3,9],bit:15,blank:0,bleu:3,block:[11,15],block_ngram_repeat:[11,15],booktitl:6,bool:[7,9,10,11],bos:11,both:[2,11,14],both_embed:14,boundari:[12,14,15],bpe:[12,14,15],bptt:[7,14],bridg:[4,18],bridge_extra_nod:14,browser:0,bucket_s:[3,14],buffer:7,build:[0,4,7,9,10,11,15,16],build_token:10,build_vocab:12,built:7,bytetensor:11,cach:9,calcul:[1,7,11],call:11,callabl:11,callback:7,can:[1,2,3,5,7,10,11,12,14,15],cancel:10,candid:[2,12,14,15],cao:18,capit:0,captur:1,cat:16,categor:11,categori:11,challeng:4,chang:[0,2,7,14],channel:1,charact:[0,15],character_coverag:16,check:[0,6,17],checklist:0,checkpoint:[3,7,14],chen:18,chmod:[3,5],choic:[0,9,12,14,15],choos:[0,12,14,15],chosen:11,citat:[0,4],cite:[0,6],classmethod:7,clear:0,clone:[6,10,17],clone_model:10,close:0,cls:7,cluster:[2,6,17],clutter:0,code:[0,2,5,15],code_dir:5,codebas:5,column:2,com:[6,17],combin:15,comma:2,command:[3,4],comment:0,commentari:3,common:[0,4],commoncrawl:3,commun:0,complet:11,complex:[2,11],compon:[1,2],composit:14,comput:[1,2,3,7,9,14,15],concat:[9,14],condit:[11,14,15],conf:[13,15],config:[3,4,10,12,13,14,15],config_fil:10,configur:[2,3,4],connect:1,consid:[2,16],consider:14,consist:0,constant:2,constructor:0,consum:14,contain:[2,9,10,11],content:[0,15],context:[1,9,14],continu:0,contribut:[0,1,9],contributor:4,control:[2,7],conv2conv:4,conveni:2,convent:0,convers:11,convert:10,copi:[0,2,4,5,14,15],copy_attn:[11,14],copy_attn_forc:14,copy_attn_typ:14,copy_loss_by_seqlength:14,core:[1,4,7],corpora:3,corpu:[2,3,12,14,16],corr:[0,18],correct:2,correspand:10,correspond:[1,15],could:11,count:[2,7,11,12,14,15],count_paramet:7,cov:11,cov_pen:11,coverag:[11,14,15],coverage_attn:14,coverage_non:11,coverage_penalti:[11,15],coverage_summari:11,coverage_wu:11,cpu:[10,14,15],crai:5,crayon:14,creat:[2,5,7],creation:2,criteria:14,criterion:7,critic:[14,15],cross:[7,14],csc:16,csv:2,ct2_model:10,ct2_translate_batch_arg:10,ct2_translator_arg:10,ctrl:0,cumbersom:2,cumul:[9,11,15],cumulative_averag:9,cumulative_average_mask:9,cur_dir:16,cur_len:11,current:[2,7,9,11,14],curricula:2,curriculum:2,custom:[10,14],custom_opt:10,cut:[0,16],cutoff:11,d_ff:9,d_model:9,dai:18,data:[1,2,4,7,11,18],data_path:16,data_typ:[7,11,14,15],dataset:[3,4,12,14,15,16],datastructur:10,dblp:0,ddress:18,deal:2,debug:[13,14,15],dec:2,dec_lay:[3,14],decai:14,decay_method:[3,14],decay_step:14,decod:[1,2,4,7],decode_strategi:11,decoder_typ:[3,14],decoderbas:7,decodestrategi:11,def:0,defin:[2,3,12,14,15],definit:9,delai:2,delet:[12,14,15],delimit:15,deng:6,denois:[2,4],denoising_object:[12,14,15],denot:1,depend:[0,2,5,7,10],deprec:[14,15],describ:[1,9,10,14],descript:0,desir:[2,3],detail:[6,12,14],determin:2,detoken:[3,10],dev:[5,16],develop:0,devic:[2,9,11,15],device_context:7,deyi:18,diagon:2,dict:[2,7,10,11,12,14,15],dict_kei:14,dictionari:[7,9,11,14],differ:[0,1,2,10,15],dim:9,dimens:[1,9,11,14],dimension:1,dir:16,direct:[0,2,11],directli:[0,15],directori:[2,5,10,14],disabl:14,discard:14,discourag:14,disk:14,displai:7,dist:7,distanc:14,distribut:[2,7,9,11,12,14,15],divers:[1,12,14,15],divid:[1,2,14,15],divis:9,do_timeout:10,doc:0,document:[0,6],doe:[2,15],doesn:16,doi:6,doing:[2,15],don:0,done:[3,11,16],dot:[1,9,14],dotprod:14,down:[11,12],download:5,dropout:[3,7,9,12,14,15],dropout_step:[3,7,14],due:14,dump:[12,14,15],dump_beam:[11,15],dump_sampl:12,dump_transform:14,dure:[10,14,15],dynam:[4,9,15],each:[1,2,9,11,12,14,15],earli:14,earlier:[1,12,14,15],early_stop:14,early_stopping_criteria:14,earlystopp:7,eas:2,easi:0,easili:2,echo:[3,16],edg:14,effect:[1,10,12],effici:[4,7,18],either:[11,14],elaps:7,elapsed_tim:7,element:[1,2],els:16,emb:9,emb_fil:9,emb_lut:9,embed:[1,4,9,12],embedding_s:9,embeddings_typ:14,emerg:1,emploi:[1,7],empti:[3,11,12,14],enabl:15,enc:2,enc_lay:[3,14],encapsul:1,encod:[1,2,4,7,11],encoder_typ:[3,14],encoderbas:7,encordec:[12,14],encount:[12,14],encout:[12,14],end:11,eng:2,english:[2,3,16],enhanc:1,ensembl:15,ensur:1,entir:16,entri:0,entropi:7,env_dir:5,environ:5,eos:11,epoch:14,epsilon:14,equal:[11,14],equat:9,equival:14,error:[0,12,14,15],especi:2,essenti:11,establish:1,eural:18,europarl:3,evalu:7,even:2,event:11,everi:[7,14,15],exactli:0,exampl:[0,2,3,12,14,17],exce:14,except:[0,10,12,14,15],exclusion_token:11,execut:[3,12,14],exist:[12,14,15,16],exp:14,exp_host:14,expect:[2,11],experi:[12,14,15],experiment:14,exponenti:14,extend:0,extern:0,extra:[5,14],extract:16,facilit:1,fail:11,fairseq:0,fals:[7,9,10,11,12,13,14,15],familiar:6,faster:14,feat_0:15,feat_1:15,feat_dim_expon:9,feat_merg:[9,14],feat_merge_s:14,feat_padding_idx:9,feat_vec_expon:[9,14],feat_vec_s:[9,14],feat_vocab_s:9,feats0:15,feats1:15,featur:[1,4,7,9,12,15,18],fed:1,feed:[2,9,14],feedforward:[1,14],feedforwardattentionbridgelay:4,feel:0,few:0,ffn:[9,14],figur:9,file:[0,2,10,12,14,15,16],filenam:14,filter:[3,4,16],filterfeat:[12,14,15],filtertoolong:[2,3,12,14,15],find:0,firefox:0,first:[0,2,9,11,14],five:1,fix:[0,11,14],flag:7,flake8:0,floattensor:[7,9,11],flow:1,fly:3,fnn:9,focu:[0,1],folder:0,follow:[0,1,2,3,15,17],foo:0,forbidden:11,forbidden_token:11,forc:[11,15],format:[0,10,12,14,15,16],forward:[2,7,9,14],fotran:2,found:16,foundat:1,fp16:[14,15],fp32:[3,7,14,15],frac:1,fraction:[12,14,15],framework:[4,14],free:[0,10],freez:[9,14],freeze_word_vec:9,freeze_word_vecs_dec:14,freeze_word_vecs_enc:14,frequenc:[12,14,15],from:[1,2,7,9,11,14,15,16],from_opt:7,frozenset:11,full:[0,2,10,12,14,15,16],full_context_align:14,fulli:2,further:[12,14],fusedadam:14,gao:18,gap:18,garg:14,gather:7,gating_output:9,gelu:14,gener:[0,1,2,3,4,7,11,15],generator_funct:14,german:3,get:[4,5],git:[6,17],github:[6,14,17],give:[2,14,15],given:[1,2,10],global_attent:14,global_attention_funct:14,global_scor:11,glove:14,gnmt:11,gnmtglobalscor:11,going:11,gold:11,gold_scor:11,gold_sent:11,gomez:18,gone:14,good:[0,14],googl:[0,11,15,18],gpu:[2,3,5,10,11,14,15],gpu_backend:14,gpu_rank:[3,14],gpu_verbose_level:[7,14],gpuid:14,grad:7,gradient:[7,14],graham:18,gram:11,graph:14,gre:5,greater:11,greedy_search:11,greedysearch:11,group:[14,15],groupwis:2,grow:11,gtx1080:15,guid:[6,17],guidelin:4,guillaum:6,had:15,haddow:18,hand:2,handl:[0,7],happen:11,has:[1,2,11,12,14,15],has_cov_pen:11,has_len_pen:11,has_tgt:11,have:[0,2,3,9,11,14,15],head:[1,3,9,14],head_count:9,help:[0,1,15],helsinki:[6,17],here:[1,11,16],hidden:[7,9,14],hidden_ab_s:14,hidden_dim:1,hieu:18,high:2,higher:[11,14,15],highest:15,hold:11,hop:1,host:5,how:0,howev:[0,7],html:[0,14],http:[1,5,6,14,15,16,17,18],huge:14,human:[2,18],hyp_:3,hyperbol:1,hyphen:2,hypothesi:3,identifi:15,idl:2,ids:2,ignor:[3,12,14,15],ignore_when_block:[11,15],illia:18,ilya:18,imag:7,impact:14,implement:[1,7,9,14],impli:1,improv:[9,11,14,18],in_config:2,includ:[0,2,9,12,14,15],incompat:[12,14,15],incorpor:14,increas:2,index:[5,9,14],indic:[1,7,9,11,12,14,15],individu:2,inf:11,infer:11,inferfeat:4,info:[14,15],inform:[1,2,14,15],ingredi:11,init:14,init_st:7,initi:[4,7,10,11],initial_accumulator_valu:14,inp:11,inp_seq_len:11,inproceed:6,input:[1,4,7,9,10,11,12,14,15,16,18],input_format:3,input_len:9,input_sentence_s:16,inputs_len:9,inputt:11,insert:[12,14,15],insert_ratio:[12,14,15],instal:[0,3,4],instanc:[7,11],instanti:7,instead:[0,2,5,12,14,15],instruct:14,int8:15,integ:11,integr:0,interact:5,interfac:7,intermedi:1,intermediate_output:1,intern:10,interv:14,introduc:[1,2],introduct:2,invalid:[12,14,15],involv:1,is_finish:11,isn:11,item:9,iter:7,its:[0,2],itself:2,jakob:18,jean:6,jinsong:18,job:5,joiner:[12,14,15],jone:18,journal:0,json:13,kaiser:18,keep:[10,11,14],keep_checkpoint:[3,14],keep_stat:14,keep_topk:11,keep_topp:11,kei:9,kera:14,key_len:9,kim:6,klau:18,klein:6,krikun:18,label:14,label_smooth:[3,14],lambda:[12,14,15],lambda_align:14,lambda_coverag:14,lang:2,lang_a:2,lang_b:2,lang_pair:[2,15],languag:[1,4,12,14,16],language_pair:16,last:[2,14,15],layer:[1,9,14,15],layer_cach:9,layer_type_to_cl:1,layernorm:14,layerstack:2,lead:11,learn:[1,7,14],learning_r:[3,7,14],learning_rate_decai:14,learning_rate_decay_fn:7,least:0,leav:[2,14],left:1,len:[7,9,11],length:[2,7,9,11,12,14,15,16],length_averag:11,length_non:11,length_pen:11,length_penalti:[11,15],length_wu:11,less:2,let:[2,3],level:[12,14],lib:5,librari:14,like:[0,11,15],limit:15,lin:[1,14],linattentionbridgelay:4,line:[0,3,12,14,15],linear:1,linear_warmup:14,linguist:[9,18],link:[0,1,5],list:[0,2,7,9,10,11,12,14,15],list_model:10,literatur:14,llion:18,load:[5,7,9,10,14],load_model:10,load_pretrained_vector:9,loader:4,local:[0,2],localhost:14,log:[4,7,11],log_fil:[14,15],log_file_level:[14,15],log_prob:11,log_tensorboard:7,logger:11,login:5,logit:[11,15],logsumexp:11,longer:15,longest:11,longtensor:[7,9,11],look:[0,6,9,15],loop:7,loss:[4,14],loss_scal:14,losscomputebas:7,love:0,lower:[2,14],lsl:[11,18],lstm:14,lua:10,lukasz:18,luong:[14,18],lustrep1:5,lustrep2:5,macherei:18,machin:[6,9,11,18],made:2,magic:11,mai:[2,7,10,11,12,14],main:[0,6,7,12,14,15],maintain:11,make:[0,5,7,12,14,15],make_shard_st:7,mammoth:[0,4,5,6,7,9,10,11,14],manag:7,mani:[7,11,14],manipul:7,manual:[10,11],map:[2,7],marian:14,mark:14,marker:10,mask:[9,12,14,15],mask_length:[12,14,15],mask_or_step:9,mask_ratio:[12,14,15],mass:[12,14,15],massiv:[2,6],master:14,master_ip:14,master_port:14,match:10,mathbb:1,mathbf:1,mathemat:1,matric:1,matrix:[1,9,14],max:[7,11,16],max_generator_batch:[3,14],max_grad_norm:[3,7,14],max_len:9,max_length:[11,15],max_relative_posit:[9,14],max_sent_length:15,max_sentence_length:16,max_siz:7,maxim:18,maximum:[12,14,15],maybe_convert_align:10,maybe_detoken:10,maybe_detokenize_with_align:10,maybe_postprocess:10,maybe_preprocess:10,maybe_token:10,maybe_update_forbidden_token:11,maybe_update_target_prefix:11,mean:[2,10,14,15],mechan:[1,2],mem:5,memori:[10,14],memory_bank:11,merg:[9,14],meta:2,metadata:7,method:[7,9,14],metric:15,mi250:5,mike:18,min_length:[11,15],minh:18,minimum:15,mirror:14,mix:7,mkdir:[5,16],mlp:[9,14],mode:[2,12,14,15],model:[1,2,4,11,12],model_dim:9,model_dtyp:[3,7,14],model_id:10,model_kwarg:10,model_prefix:16,model_root:10,model_sav:7,model_step:3,model_task:14,model_typ:14,modelsaverbas:7,modif:7,modifi:[0,11],modul:[0,1,4,5,7,14,15],modular:6,mohammad:18,monolingu:2,more:[0,2,11,12,14,15],most:[11,15],mostli:7,move:[10,14],moving_averag:[7,14],much:14,multi:[0,1,9],multiheadedattent:[1,9],multilingu:[2,6],multipl:[0,1,2,7,9,14,15],multipli:1,multplic:0,must:[2,9,10,14],mymodul:5,n_batch:7,n_best:[10,11,15],n_bucket:14,n_correct:7,n_edge_typ:14,n_node:14,n_sampl:[3,12,14],n_seg:10,n_src_word:7,n_step:14,n_word:7,name:[0,2,4,11,12,14,16],namespac:10,napoleon:0,nccl:14,necessari:[0,3,5,7,11,14,15],necessit:2,need:[0,2,3,7,9,14,18],neg:[10,14],network:[9,18],neubig:18,neural:[6,9,11,18],never:11,news_commentari:3,next:[2,7,11,15],nfeat:9,ngram:[11,15],nightmar:2,niki:18,nlp:[6,17],nmt:[7,11,14,15],nmtmodel:7,noam:[3,14,18],noamwd:14,node:[2,5,7,14],node_rank:14,nois:2,non:[9,11,14],none:[7,9,10,11,12,14,15],nonetyp:[9,11],norm:[9,14],norm_method:7,normal:[1,3,7,14],normalz:7,norouzi:18,note:[0,2,3,5,11],noth:[0,7],notset:[14,15],ntask:5,nucleu:15,num_step:7,num_thread:12,number:[1,2,7,9,11,12,14,15],nvidia:14,obj:[0,7],object:[0,7,10,11,12,14,15,16],oder:2,off:14,ofi:5,often:[12,14,15],on_timemout:10,on_timeout:10,onc:[11,14],one:[0,1,2,7,12,14,15],onli:[2,7,11,12,14,15],onmt:16,onmt_build_vocab:3,onmt_token:[12,14,15],onmt_transl:3,onmttok:4,open:6,opennmt:[0,2,5,6,7,13],oper:1,operatornam:1,opt:[3,7,10,14,15],opt_level:14,optim:[3,4],option:[0,2,3,5,7,9,10,11,12,14,15,16],opu:4,opus100:[16,17],ord:18,order:[2,14],org:[1,5,6,14,15,18],origin:[1,14,16],oriol:18,other:[1,5,7,11,12,14,15,16,18],other_lang:16,otherwis:[2,9,14,15],our:[5,11],our_stat:7,out:[1,2,6,7,17],out_config:2,out_fil:11,outcom:1,output:[1,2,3,7,9,10,11,12,14,15],output_model:15,over:[0,2,3,7,11,14,15,16],overal:1,overrid:[11,12,14],overview:4,overwrit:[5,12,14],own:[7,15],ownership:7,p17:6,p18:14,packag:[5,10],pad:[7,9,11],pair:[2,7,10,14,15,16],paper:[0,1,14],parallel:[9,11,12,14],parallel_path:11,parallelcorpu:11,param:7,param_init:[3,14],param_init_glorot:[3,14],paramet:[3,7,9,10,11,12,14,15],parenthes:0,parmar:18,pars:10,parse_opt:10,part:[1,11],particular:[0,2],partit:5,pass:[1,2,7,10,14],past:[0,14],path:[2,5,9,10,11,12,14,15],path_src:3,path_tgt:3,patienc:7,pattern:2,pdf:14,pen:11,penalti:[4,11,14],penaltybuild:11,peopl:5,per:[0,2,12,14,15],perceiv:[1,14],perceiverattentionbridgelay:4,percentag:[12,14,15],perfom:14,perform:[1,14],permut:[12,14,15],permute_sent_ratio:[12,14,15],perplex:7,pfs:5,pham:18,phrase_t:[11,15],piec:3,pip3:[5,6,17],pip:[0,5],pipelin:[12,14,15],pleas:[0,6],plu:14,poisson:[12,14,15],poisson_lambda:[12,14,15],polosukhin:18,pool:14,port:[13,14],portal:6,pos_ffn_activation_fn:[9,14],posit:[9,14],position_encod:[9,14],position_ffn:9,positionalencod:9,positionwisefeedforward:[9,14],possibl:[2,7,10,11,12,14,15],postprocess:10,postprocess_opt:10,potenti:11,pouta:16,ppl:7,pre:[7,10,11],pre_word_vecs_dec:14,pre_word_vecs_enc:14,preced:2,precis:7,pred:15,pred_scor:11,pred_sent:11,predict:[7,11,15],prefer:0,prefix:[2,7,12,14,15],prefix_seq_len:11,preliminari:3,preload:10,preload_model:10,prepar:[4,11],prepare_wmt_data:3,preprint:18,preprocess:10,preprocess_opt:10,presenc:2,presum:11,pretrain:[9,14],prevent:[11,15],previou:[1,2,9,11],previous:1,primari:2,prime:1,print:[7,14,15],prior:3,prior_token:[12,14,15],prob:11,proba:15,probabl:[9,11,12,14,15],problem:11,proc:[6,18],procedur:2,process:[1,7,10,12,14],processu:10,produc:[1,11,12,14,15],product:1,projappl:5,project:[0,1,5,6],project_2005099:5,project_462000125:5,propag:7,proper:10,properli:5,properti:[7,9],proport:[2,12,14,15],provid:[6,15],prune:4,pty:5,pull_request_chk:0,punctuat:0,put:11,pwd:16,pyonmttok:[12,14,15],python3:[2,5],python:[0,2,5,14],pythonpath:5,pythonuserbas:5,pytorch:[0,5],qin:18,quantiz:15,queri:9,query_len:9,queue:[12,14],queue_siz:[3,14],quickstart:[4,6],quoc:18,quot:0,rais:[12,14],random:[4,12,14],random_ratio:[12,14,15],random_sampling_temp:[11,15],random_sampling_topk:[11,15],random_sampling_topp:[11,15],randomli:11,rang:15,rank:[11,14],ranslat:18,rare:11,rate:[4,7],rather:0,ratio:[11,15],raw:[11,15],rccl:5,reach:11,read:[0,2,10,16],readabl:[0,2],reader:4,readm:14,rebuild:10,rebuild_seg_packag:10,receiv:2,recent:14,recommend:14,recommonmark:0,rectifi:1,recurr:9,redund:2,ref:0,refer:[0,1,4],regardless:2,regular:[12,14,15],rel:14,relat:[3,12,14,15],relationship:1,relev:[9,11],relu:[1,9,14],rememb:0,remov:2,renorm:14,reorder:11,repeat:[11,15],repetit:15,replac:[11,12,14,15],replace_length:[12,14,15],replace_unk:[11,15],report:[6,7,14,15],report_align:[11,15],report_everi:[3,14],report_manag:7,report_scor:11,report_stats_from_paramet:[7,14],report_tim:[11,15],reportmgrbas:7,repres:[1,7],represent:[1,14],reproduc:4,requir:[0,7,14],research:6,reset:7,reset_optim:14,resett:14,residu:9,resourc:2,respect:[1,2],respons:7,rest:13,restrict:[12,14,15],result:[1,10,14],return_attent:11,reus:14,reuse_copy_attn:14,revers:[12,14,15],reversible_token:[12,14,15],rico:18,right:[0,1],rmsnorm:14,rnn:[7,14],rnn_size:[3,14],roblem:18,rocm5:5,rocm:5,root:[1,2],rotat:[12,14,15],rotate_ratio:[12,14,15],roundrobin:14,row:2,rsqrt:14,rst:0,run:[0,2,3,7,10,14,15],rush:6,sacrebleu:[3,5,6,17],sai:2,samantao:5,same:[0,2,3,9,10,14],sampl:[4,11,12,14,16],sample_with_temperatur:11,sampling_temp:11,saniti:15,save:[7,12,14,15,16],save_all_gpu:14,save_checkpoint_step:[3,7,14],save_config:[12,14,15],save_data:[3,12,14],save_model:[3,14],saver:7,scale:[11,14],schedul:[7,14],schuster:18,score:[4,10,15],scorer:11,scratch:5,script:[0,3,4,5],search:[0,2,4,11],second:[1,2,9,10],secur:[12,14],see:[2,9,10,11,12,14],seed:[3,11,12,14,15],seemingli:14,seen:1,segment:[2,10,15],select:[9,11],select_index:11,self:[1,9,10,11,14],self_attn_typ:14,send:[0,14],senellart:6,sennrich:18,sensibl:0,sent:[7,14,15],sent_numb:11,sentenc:[11,12,14,15,16],sentencepiec:[2,3,5,6,12,14,15,17],separ:2,seper:10,seq2seq:[11,14],seq:11,seq_len:[1,9,11],sequenc:[1,2,7,9,10,11,12,14,15],serial:9,serv:1,server:[4,14,16],servermodel:10,servermodelerror:10,session:5,set:[1,2,3,5,7,9,10,11,12,14,15],setup:3,sever:[2,9,11],sgd:14,sh16:[9,18],shape:[0,9,11],shard:[7,14,15],shard_siz:[7,15],share:[5,12,14,15],share_decoder_embed:[3,14],share_embed:[3,14],share_vocab:[12,14],shazeer:18,shortest:11,shot:2,should:[2,3,11,14],shuf:16,shuffle_input_sent:16,side:[2,7,10,12,14,15],side_a:2,side_b:2,silent:[3,12,14],similar:[1,2,9,14],simpl:[1,7,14],simpleattentionbridgelay:4,simulatan:9,sin:14,singl:[0,10,14],single_pass:14,sinusoid:9,site:5,size:[2,7,9,11,12,14,15,16],skip:[2,12,14],skip_empty_level:[3,12,14],slow:[12,15],slurm:[2,5],smaller:[12,14,15],smooth:[12,14,15],softmax:[1,14,15],some:[0,2,7,15],someth:0,sometim:0,sort:[10,16],sorted_pair:2,sourc:[0,2,4,5,6,7,9,10,11,12,14],sp_path:16,space:[0,1,14],spacer:[12,14,15],span:[12,14,15],specif:[1,2,6,11,12,14,17],specifi:[1,12,14,15],sphinx:0,sphinx_rtd_them:0,sphinxcontrib:0,spill:0,spm_decod:3,spm_encod:[3,16],spm_train:16,sqrt:1,squar:[1,2],src:[2,3,7,10,11,12,14,15,16],src_embed:14,src_feat:15,src_feats_vocab:[12,14],src_file_path:11,src_ggnn_size:14,src_group:2,src_lang:[2,15],src_languag:2,src_len:7,src_length:11,src_map:11,src_onmttok_kwarg:[12,14,15],src_raw:11,src_seq_length:[3,12,14,15],src_seq_length_trunc:14,src_subword_alpha:[3,12,14,15],src_subword_model:[3,12,14,15],src_subword_nbest:[3,12,14,15],src_subword_typ:[12,14,15],src_subword_vocab:[12,14,15],src_vocab:[3,11,12,14],src_vocab_s:14,src_vocab_threshold:[12,14,15],src_word_vec_s:14,src_words_min_frequ:14,sru:4,srun:5,stabl:1,stack:[14,15],stage:1,stand:0,standard:[9,14,15],start:[2,4,5,7,10,14,16],start_decay_step:14,stat:[7,14],stat_list:7,state:[7,11,14],state_dict:14,state_dim:14,statist:[7,14],stdout:7,step:[1,2,4,7,9,11,14,15],stepwis:9,stepwise_penalti:[11,15],still:0,stop:[12,14,15],store:14,str:[0,7,9,10,11],strategi:[4,7,14],string:[7,9,12,14,15],structur:[1,4],style:[0,12,14,15],styleguid:0,subclass:[7,11],subcompon:2,subdirectori:5,subsequ:1,subset:16,substitut:2,subword:[2,4],suggest:14,sum:[7,9,11,14],sume:7,summari:[0,11,15],superclass:0,supervis:[2,14],support:[0,2,14],suppos:16,sure:[5,11],sutskev:18,switchout:[4,18],switchout_temperatur:[12,14,15],symmetr:2,system:[11,14,18],tab:[12,14],tabl:[9,15],take:[1,2,6,9,12,14,15],tangent:1,tanh:1,tar:16,target:[2,4,7,10,11,12,14],target_prefix:11,task:[2,3,4,7,11],task_distribution_strategi:14,task_queue_manag:7,tatoeba:[2,4],tau:[12,14,15],technic:6,temperatur:[2,11,12,14,15],templat:2,tensor:[0,7,9,11],tensorboard:[7,14],tensorboard_log_dir:14,tensorflow:14,term:1,test:[0,3,5,9],testset:3,text:[7,11,14,15],tgt:[2,3,7,10,12,14,15],tgt_embed:14,tgt_file_path:11,tgt_group:2,tgt_lang:[2,15],tgt_languag:2,tgt_len:7,tgt_onmttok_kwarg:[12,14,15],tgt_prefix:[11,15],tgt_sent:11,tgt_seq_length:[3,12,14,15],tgt_seq_length_trunc:14,tgt_subword_alpha:[3,12,14,15],tgt_subword_model:[3,12,14,15],tgt_subword_nbest:[3,12,14,15],tgt_subword_typ:[12,14,15],tgt_subword_vocab:[12,14,15],tgt_vocab:[3,7,12,14],tgt_vocab_s:14,tgt_vocab_threshold:[12,14,15],tgt_word_vec_s:14,tgt_words_min_frequ:14,than:[0,11,14,16],thang:18,thant:11,thei:[1,11],them:2,thi:[0,1,2,3,5,6,7,9,11,12,14,15],thin:7,thing:[0,2],thoroughli:9,thread:12,three:1,through:[1,2,7],thu:7,tic:0,tick:0,time:[1,2,5,7,11,14,15],timeout:10,timer:10,titl:6,to_cpu:10,to_gpu:10,todo:[5,16],tok:10,token:[3,7,10,11,12,14,15],token_drop:4,token_mask:4,tokendrop:[12,14,15],tokendrop_temperatur:[12,14,15],tokenizer_mark:10,tokenizer_opt:10,tokenmask:[12,14,15],tokenmask_temperatur:[12,14,15],too:11,tool:4,toolkit:6,top:[1,11,15],topk_id:11,topk_scor:11,torch:[0,5,7,9,14],torchtext:7,total:[2,7,14],trail:0,train:[2,4,5,6,7,9],train_extremely_large_corpu:16,train_from:14,train_it:7,train_loss:7,train_loss_md:7,train_step:[3,7,14],trainabl:7,trainer:4,training_step:7,transform:[1,3,4,7,18],transformer_ff:[3,14],transformerattentionbridgelay:4,transformerencoderlay:1,translat:[2,4,6,7,9,10,13,18],translate_batch:11,translation_serv:10,translationbuild:11,translationserv:10,travi:0,trg:2,triang:2,trick:[4,9],trunc_siz:7,truncat:[7,14],truncated_decod:14,trust:16,turn:14,tutori:[4,17],two:[1,2,9],txt:[0,15,16],type:[0,1,2,4,7,9,10,11,12,15],typic:[7,14],under:[2,14,15],undergo:1,undergon:1,underli:11,uniform:14,unigram:[12,14,15],union:0,unit:1,unittest:0,unk:[11,15],unknown:11,unless:2,unload:10,unload_model:10,unmodifi:11,unnecessari:[0,2],unset:2,until:[11,15],unwieldli:2,updat:[5,7,10,11,14],update_dropout:9,update_finish:11,update_learning_r:14,update_n_src_word:7,update_vocab:14,upgrad:5,upper:2,url:[5,6,18],url_root:13,usag:[4,12,13,14,15],use:[0,1,2,3,5,7,9,10,11,12,14,15,16],used:[1,2,3,7,9,10,11,12,14,15],useful:7,user:[5,7,9,10],uses:[0,2,9,11,14],using:[0,1,2,6,9,10,11,12,14,15],uszkoreit:18,util:[1,7],v11:3,valid:[3,7,12,14,15],valid_batch_s:[3,14],valid_it:7,valid_loss:7,valid_loss_md:7,valid_step:[3,7,14],valu:[1,2,7,9,10,11,12,14,15],variabl:[2,5,11],variat:0,vaswani:18,vaswanispujgkp17:0,vector:[9,14],venv:5,verbos:[11,14,15],veri:[0,15],version:[10,11],via:[9,18],vinyal:18,virtual:5,visit:0,visual:14,vocab:[3,4,7,11],vocab_path:[12,14],vocab_s:[11,14,16],vocab_sample_queue_s:12,vocab_size_multipl:14,vocabulari:[2,7,12,14,15,16],vsp:[9,18],wai:[2,11],wait:2,wang:18,want:[2,15],warmup:14,warmup_step:[3,14],warn:[12,14,15],weight:[1,2,3,9,14,15],weight_decai:14,weighted_sampl:14,well:[0,14],wget:16,what:[2,7,10],when:[0,2,6,9,11,12,14,15,16],where:[1,3,5,9,11,12,14,15],wherea:[11,14],whether:[7,10,11,12,14,15],which:[2,9,11,14],whl:5,whole:[3,11],whose:15,why:1,wiki:14,wikipedia:14,window:[12,14,15],wise:1,with_align:7,within:[1,10],without:[0,14],wmt14_en_d:3,wmt:3,wmtend:3,wojciech:18,wolfgang:18,word2vec:14,word:[1,9,11,12,14,15],word_align:11,word_lut:9,word_padding_idx:9,word_vec_s:[3,9,14],word_vocab_s:9,work:[0,2,11,14],workflow:6,world_siz:[3,14],would:[2,11,14],wpdn18:[12,14,15,18],wrap:10,wrapper:7,writabl:2,write:[2,7],writer:7,written:3,wsc:[11,18],www:14,xavier_uniform:14,xent:7,xinyi:18,xiong:18,xzvf:16,yaml:[3,12,14,15],year:6,yet:11,yml:0,yonghui:18,yoon:6,you:[0,2,3,5,9,14,15,18],your:[0,2,5,15,16],your_venv_nam:5,your_vevn_nam:5,yourself:6,yuan:18,yuntian:6,zaremba:18,zero:[2,7,9,11,14,15],zero_grad:7,zhang:18,zhifeng:18,zihang:18,zxs18:[9,18]},titles:["Contributors","Attention Bridge","Config-config tool","Translation","Contents","Installation","Overview","Framework","Data Loaders","Modules","Server","Translation","Build Vocab","Server","Train","Translate","Prepare Data","Quickstart","References"],titleterms:{"class":11,The:2,actual:2,adapt:[2,14],adapter_config:2,ae_path:2,ae_transform:2,align:14,allocate_devic:2,altern:2,architectur:9,argument:13,attent:[1,9,14],autoencod:2,beam:15,bridg:[1,14],build:[3,12],challeng:16,citat:6,cluster_languag:2,command:2,common:[12,14,15],complete_language_pair:2,config:2,config_al:2,config_config:2,configur:[12,14,15],content:4,contributor:0,conv2conv:9,copi:9,core:[9,10],corpora:2,corpora_schedul:2,data:[3,8,12,14,15,16,17],dataset:8,dec_sharing_group:2,decod:[9,11,14,15],denois:[12,14,15],direct:16,distanc:2,distance_matrix:2,docstr:0,download:[3,16],dynam:14,effici:15,embed:14,enc_sharing_group:2,encod:[9,14],evalu:3,featur:14,feedforwardattentionbridgelay:1,filter:[12,14,15],framework:7,gener:14,get:16,group:2,guidelin:0,inferfeat:[12,14,15],initi:14,input:2,instal:[5,6,17],kei:2,languag:[2,15],level:2,linattentionbridgelay:1,line:2,loader:8,log:[14,15],loss:7,lumi:5,mahti:5,mammoth:17,manual:2,matrix:2,model:[3,7,10,14,15,16],modul:9,n_gpus_per_nod:2,n_group:2,n_node:2,name:13,onmttok:[12,14,15],optim:[7,14],opu:16,other:2,overrid:2,overview:6,paramet:2,pars:16,path:16,penalti:15,perceiverattentionbridgelay:1,prepar:[3,16,17],prune:14,puhti:5,quickstart:17,random:15,rate:14,reader:8,refer:18,relev:16,remove_temporary_kei:2,reproduc:[12,14,15],run:5,sampl:15,score:11,search:15,sentencepiec:16,server:[10,13],set:16,set_transform:2,share:2,sharing_group:2,shot:16,simpleattentionbridgelay:1,sourc:15,specifi:2,src_path:2,sru:9,stage:2,step:[3,16,17],strategi:11,structur:9,subword:[3,12,14,15],supervis:16,switchout:[12,14,15],target:15,task:14,tatoeba:16,test:16,tgt_path:2,than:2,token_drop:[12,14,15],token_mask:[12,14,15],tool:2,top:2,train:[3,14,16],trainer:7,transform:[2,9,12,14,15],transformerattentionbridgelay:1,translat:[3,11,15,16],translation_config:2,translation_config_dir:2,trick:15,type:14,usag:2,use_introduce_at_training_step:2,use_weight:2,valid:16,variabl:16,vocab:[12,14,16],vocabulari:3,yaml:2,zero:16,zero_shot:2}})