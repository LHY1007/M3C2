# from apex import amp
from utils import *
import dataset_mine
from net import init_weights,get_scheduler,WarmupCosineSchedule
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    if seed == 0:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def train(opt):
    opt['gpus'] = [0,1]
    gpuID = opt['gpus']
    opt['batchSize']=4
    ############### Mine_model #######################
    Mine_model_init,Mine_model_IDH,Mine_model_1p19q,Mine_model_CDKN,Mine_model_Graph,Mine_model_His,Mine_model_Cls,Mine_model_Task\
        ,opt_init,opt_IDH,opt_1p19q,opt_CDKN,opt_Graph,opt_His,opt_Cls,opt_Task=get_model(opt)
    
    if opt['decayType']=='exp' or opt['decayType']=='step':
        Mine_model_sch_init = get_scheduler(opt_init, opt['n_ep'], opt['n_ep_decay'], opt['decayType'], -1)
        Mine_model_sch_IDH = get_scheduler(opt_IDH, opt['n_ep'], opt['n_ep_decay'], opt['decayType'], -1)
        Mine_model_sch_1p19q = get_scheduler(opt_1p19q, opt['n_ep'], opt['n_ep_decay'], opt['decayType'], -1)
        Mine_model_sch_CDKN = get_scheduler(opt_CDKN, opt['n_ep'], opt['n_ep_decay'], opt['decayType'], -1)
        Mine_model_sch_Graph = get_scheduler(opt_Graph, opt['n_ep'], opt['n_ep_decay'], opt['decayType'], -1)
        Mine_model_sch_His = get_scheduler(opt_His, opt['n_ep'], opt['n_ep_decay'], opt['decayType'], -1)
        Mine_model_sch_Cls = get_scheduler(opt_Cls, opt['n_ep'], opt['n_ep_decay'], opt['decayType'], -1)
        Mine_model_sch_Task = get_scheduler(opt_Task, opt['n_ep'], opt['n_ep_decay'], opt['decayType'], -1)
    elif opt['decayType']=='cos':
        Mine_model_sch_init = WarmupCosineSchedule(opt_init, warmup_steps=opt['decay_cos_warmup_steps'], t_total=opt['n_ep'])
        Mine_model_sch_IDH = WarmupCosineSchedule(opt_IDH, warmup_steps=opt['decay_cos_warmup_steps'], t_total=opt['n_ep'])
        Mine_model_sch_1p19q = WarmupCosineSchedule(opt_1p19q, warmup_steps=opt['decay_cos_warmup_steps'], t_total=opt['n_ep'])
        Mine_model_sch_CDKN = WarmupCosineSchedule(opt_CDKN, warmup_steps=opt['decay_cos_warmup_steps'], t_total=opt['n_ep'])
        Mine_model_sch_Graph = WarmupCosineSchedule(opt_Graph, warmup_steps=opt['decay_cos_warmup_steps'],t_total=opt['n_ep'])
        Mine_model_sch_His = WarmupCosineSchedule(opt_His, warmup_steps=opt['decay_cos_warmup_steps'], t_total=opt['n_ep'])
        Mine_model_sch_Cls = WarmupCosineSchedule(opt_Cls, warmup_steps=opt['decay_cos_warmup_steps'],t_total=opt['n_ep'])
        Mine_model_sch_Task = WarmupCosineSchedule(opt_Task, warmup_steps=opt['decay_cos_warmup_steps'],t_total=opt['n_ep'])

    print('%d GPUs are working with the id of %s' % (torch.cuda.device_count(), str(gpuID)))


    root =r'/home/hanyu/LHY/miccai7.22/best_model/noGraph_0727-1651/Mine_model-0009.pth'
    ckptdir = os.path.join(root)
    checkpoint = torch.load(ckptdir)


    related_params = {k: v for k, v in checkpoint['init'].items()}
    Mine_model_init.load_state_dict(related_params)
    related_params = {k: v for k, v in checkpoint['IDH'].items()}
    Mine_model_IDH.load_state_dict(related_params)
    related_params = {k: v for k, v in checkpoint['1p19q'].items()}
    Mine_model_1p19q.load_state_dict(related_params)
    related_params = {k: v for k, v in checkpoint['CDKN'].items()}
    Mine_model_CDKN.load_state_dict(related_params)
    related_params = {k: v for k, v in checkpoint['Graph'].items()}
    Mine_model_Graph.load_state_dict(related_params)
    related_params = {k: v for k, v in checkpoint['His'].items()}
    Mine_model_His.load_state_dict(related_params)
    related_params = {k: v for k, v in checkpoint['Cls'].items()}
    Mine_model_Cls.load_state_dict(related_params)
    related_params = {k: v for k, v in checkpoint['Task'].items()}
    Mine_model_Task.load_state_dict(related_params)
    ###############  Datasets #######################

    trainDataset = dataset_mine.Our_Dataset(phase='Train',opt=opt)
    # valDataset = dataset_mine.Our_Dataset(phase='Val', opt=opt)
    testDataset = dataset_mine.Our_Dataset(phase='Test',opt=opt)
    ItestDataset = dataset_mine.Our_Dataset(phase='ITest',opt=opt)
    trainLoader = DataLoader(trainDataset, batch_size=opt['batchSize'],
                             num_workers=opt['nThreads'] if (sysstr == "Linux") else 1, shuffle=True)
    # valLoader = DataLoader(valDataset, batch_size=opt['Val_batchSize'],
    #                          num_workers=opt['nThreads'] if (sysstr == "Linux") else 1, shuffle=False)
    testLoader = DataLoader(testDataset, batch_size=opt['Test_batchSize'],
                            num_workers=opt['nThreads'] if (sysstr == "Linux") else 1, shuffle=False)
    ItestLoader = DataLoader(ItestDataset, batch_size=opt['Test_batchSize'],
                            num_workers=opt['nThreads'] if (sysstr == "Linux") else 1, shuffle=False)

    ############## initialize #######################

    last_ep = 0
    total_it = 0

    saver = Saver(opt)
    print('%d epochs and %d iterations has been trained' % (last_ep, total_it))
    alleps = opt['n_ep'] - last_ep
    allit = len(trainLoader)

    ############# begin training ##########################
    for epoch in range(alleps):

        Mine_model_sch_init.step()
        Mine_model_sch_IDH.step()
        Mine_model_sch_1p19q.step()
        Mine_model_sch_CDKN.step()
        # Mine_model_sch_Graph.step()
        Mine_model_sch_His.step()
        Mine_model_sch_Cls.step()
        Mine_model_sch_Task.step()
        Mine_model_init.train()
        Mine_model_IDH.train()
        Mine_model_1p19q.train()
        Mine_model_CDKN.train()
        Mine_model_Graph.eval()###############noGraph
        Mine_model_His.train()
        Mine_model_Cls.train()
        Mine_model_Task.train()
        curep = last_ep + epoch
        lossdict = {'train/init_subnet': 0,'train/IDH_subnet': 0,  'train/1p19q_subnet': 0,'train/CDKN_subnet': 0,'train/Graph_subnet': 0,  'train/His_subnet': 0,'train/Grade_subnet': 0,'train/Cls_subnet': 0,'train/Task_subnet': 0}
        count=0
        running_results = {'acc_IDH': 0,'acc_1p19q': 0,'acc_CDKN': 0,'acc_His': 0,'acc_Grade': 0,'acc_Diag': 0,'acc_Task': 0,
                           'loss_IDH': 0,'loss_1p19q': 0,'loss_CDKN': 0,'loss_His': 0,'loss_Grade': 0,'loss_Diag': 0,'loss_Task': 0}
        train_bar = tqdm(trainLoader)
        for packs in train_bar:
            features20x,features10x, label = packs
            count+=1
            if  torch.cuda.is_available():
                features20x = features20x.cuda(gpuID[0])
                features10x = features10x.cuda(gpuID[0])
                label = label.cuda(gpuID[0])
            label_IDH=label[:,0]
            label_1p19q = label[:, 1]
            label_CDKN = label[:, 2]
            label_His = label[:, 3]
            label_Grade = label[:, 4]
            label_Diag = label[:, 5]
            label_His_2class= label[:, 6]
            ### ### forward IDH
            # init_feature=Mine_model_init(features20x,features10x) # (BS,2500,1024)
            init_feature_his, init_feature_mark, His, Mark, Pub_His, Pub_Mark = Mine_model_init(features20x,features10x)

            hidden_states, encoded_IDH = Mine_model_IDH(init_feature_mark)
            hidden_states, encoded_1p19q = Mine_model_1p19q(hidden_states)
            encoded_CDKN = Mine_model_CDKN(hidden_states)
            results_dict,weight_IDH_wt,weight_IDH_mut,weight_1p19q_codel,weight_CDKN_HOMDEL,encoded_IDH0,encoded_1p19q0,encoded_CDKN0,Mark_output = Mine_model_Graph(encoded_IDH, encoded_1p19q, encoded_CDKN)

            pred_IDH=results_dict['logits_IDH']
            pred_1p19q = results_dict['logits_1p19q']
            pred_CDKN = results_dict['logits_CDKN']

            ### ### backward IDH
            Mine_model_CDKN.zero_grad()
            # Mine_model_Graph.zero_grad()
            Mine_model_1p19q.zero_grad()
            Mine_model_IDH.zero_grad()
            Mine_model_init.zero_grad()
            loss_1p19q = Mine_model_Graph.module.calculateLoss_1p19q(pred_1p19q, label_1p19q.type(torch.long))
            loss_CDKN = Mine_model_Graph.module.calculateLoss_CDKN(pred_CDKN, label_CDKN.type(torch.long))
            loss_IDH = Mine_model_Graph.module.calculateLoss_IDH(pred_IDH, label_IDH.type(torch.long))
            loss_Graph = Mine_model_Graph.module.calculateLoss_Graph(encoded_IDH0,encoded_1p19q0,encoded_CDKN0)

            loss_IDH_subnet = loss_IDH + 0.6*loss_1p19q + 0.3*loss_CDKN#+ 0.1 * loss_Graph
            loss_IDH_subnet.backward()
            opt_init.step()
            opt_IDH.step()
            opt_1p19q.step()
            opt_CDKN.step()
            # opt_Graph.step()

            ### ### forward His
            init_feature_his, _, _, _, _, _ = Mine_model_init(features20x,features10x)
            hidden_states, encoded_His = Mine_model_His(init_feature_his)
            results_dict, weight_His_GBM, weight_His_GBM_Cls2,weight_His_O,His_output = Mine_model_Cls(encoded_His)
            pred_His = results_dict['logits_His']
            pred_His_2class = results_dict['logits_His_2class']
            pred_Grade = results_dict['logits_Grade']


            ### ### backward His
            Mine_model_Cls.zero_grad()
            Mine_model_His.zero_grad()
            Mine_model_init.zero_grad()
            loss_His = Mine_model_His.module.calculateLoss_His(pred_His, label_His.type(torch.long))
            loss_His_2class = Mine_model_His.module.calculateLoss_His_2class(pred_His_2class, label_His_2class.type(torch.long))
            #loss_Grade = Mine_model_His.module.calculateLoss_Grade(pred_Grade, label_Grade.type(torch.long))

            loss_His_subnet = 3*loss_His_2class+loss_His
            # loss_His_subnet = loss_Grade
            loss_His_subnet.backward()
            opt_init.step()
            opt_His.step()
            # opt_Grade.step()
            opt_Cls.step()

            ##########################################
            if count %2==0:
                Mine_model_init.zero_grad()
                __, __, His, Mark, Pub_His, Pub_Mark = Mine_model_init(features20x,features10x)

                c_loss = Mine_model_init.module.calculateLoss_init(His, Mark, Pub_His, Pub_Mark)

                c_loss.backward()
                opt_init.step()
            ##########################################

            # omic mutual correlation
            # forward
            if count %20==0:
                Mine_model_Cls.zero_grad()
                Mine_model_His.zero_grad()
                Mine_model_init.zero_grad()
                Mine_model_CDKN.zero_grad()
                # Mine_model_Graph.zero_grad()
                Mine_model_1p19q.zero_grad()
                Mine_model_IDH.zero_grad()
                # init_feature=Mine_model_init(features20x,features10x)  # (BS,2500,1024)
                init_feature_his, init_feature_mark, __, __, __, __ =Mine_model_init(features20x,features10x)  # (BS,2500,1024)
                hidden_states, encoded_IDH = Mine_model_IDH(init_feature_mark)
                hidden_states, encoded_1p19q = Mine_model_1p19q(hidden_states)
                encoded_CDKN = Mine_model_CDKN(hidden_states)
                results_dict,weight_IDH_wt,weight_IDH_mut,weight_1p19q_codel,weight_CDKN_HOMDEL,encoded_IDH0,encoded_1p19q0,encoded_CDKN0,Mark_output = Mine_model_Graph(encoded_IDH, encoded_1p19q, encoded_CDKN)
                pred_IDH = results_dict['logits_IDH']
                pred_1p19q = results_dict['logits_1p19q']
                pred_CDKN = results_dict['logits_CDKN']

                hidden_states, encoded_His = Mine_model_His(init_feature_his)
                results_dict, weight_His_GBM, weight_His_GBM_Cls2,weight_His_O,His_output = Mine_model_Cls(encoded_His)
                pred_His_2class = results_dict['logits_His_2class']

                ####### recently added--convert to 2500
                for i in range(weight_IDH_wt.detach().cpu().numpy().shape[0]):
                    if i==0:
                        loss_mutual_correlation =  Mine_model_Cls.module.Loss_mutual_correlation(
                            weight_IDH_wt[i], weight_His_GBM[i], weight_1p19q_codel[i],weight_His_O[i], epoch)
                    else:
                        loss_mutual_correlation+=Mine_model_Cls.module.Loss_mutual_correlation(
                            weight_IDH_wt[i], weight_His_GBM[i], weight_1p19q_codel[i],weight_His_O[i], epoch)
                loss_mutual_correlation=opt['Network']['corre_loss_ratio']*loss_mutual_correlation/(opt['batchSize'])
                ####### recently added--convert to 2500
                loss_mutual_correlation.requires_grad_(True)
                loss_mutual_correlation.backward()

                opt_init.step()
                opt_His.step()
                opt_Cls.step()
                opt_IDH.step()
                opt_1p19q.step()
                opt_CDKN.step()
                # opt_Graph.step()

            Mine_model_Cls.zero_grad()
            Mine_model_His.zero_grad()
            Mine_model_init.zero_grad()
            Mine_model_CDKN.zero_grad()
            # Mine_model_Graph.zero_grad()
            Mine_model_1p19q.zero_grad()
            Mine_model_IDH.zero_grad()
            Mine_model_Task.zero_grad()
            init_feature_his, init_feature_mark, __, __, __, __ =Mine_model_init(features20x,features10x)

            hidden_states, encoded_IDH = Mine_model_IDH(init_feature_mark)
            hidden_states, encoded_1p19q = Mine_model_1p19q(hidden_states)
            encoded_CDKN = Mine_model_CDKN(hidden_states)
            results_dict,weight_IDH_wt,weight_IDH_mut,weight_1p19q_codel,weight_CDKN_HOMDEL,encoded_IDH0,encoded_1p19q0,encoded_CDKN0, Mark_output= Mine_model_Graph(encoded_IDH, encoded_1p19q, encoded_CDKN)
            hidden_states, encoded_His = Mine_model_His(init_feature_his)# encoded_His[BS,2500,512]
            results_dict, weight_His_GBM, weight_His_GBM_Cls2,weight_His_O,His_output = Mine_model_Cls(encoded_His)
            #
            his_mark = Mine_model_Task(His_output.float(), Mark_output.float())
            

            #[BS,512],#[BS,512],#[BS,6]
            
            weight_size = Mine_model_Task.module.fc_out.weight.size(1)
            
            loss_Task= Mine_model_Task.module.calculateLoss_Task(his_mark,label_Diag.type(torch.long))
            loss_Task.backward()
            #####################################
            if count %10==0:
                g_mark = Mine_model_Task.module.fc_out.weight.grad.clone()[:, weight_size // 2:]
                g_his = Mine_model_Task.module.fc_out.weight.grad.clone()[:, :weight_size // 2]
            
                # 调制
                _, predicted_His_2class = torch.max(pred_His_2class.data, 1)
                g_mark, g_his = adjust_gradients(g_mark, g_his, predicted_His_2class,lambda_reg=0.5)
                
                # 更新
                Mine_model_Task.module.fc_out.weight.grad[:, weight_size // 2:] = g_mark
                Mine_model_Task.module.fc_out.weight.grad[:, :weight_size // 2] = g_his
            ###############################
            opt_init.step()
            opt_His.step()
            opt_Cls.step()
            opt_IDH.step()
            opt_1p19q.step()
            opt_CDKN.step()
            # opt_Graph.step()
            opt_Task.step()

            _, predicted_Task = torch.max(his_mark.data, 1)
            total_Task = label_Diag.size(0)
            correct_Task = predicted_Task.eq(label_Diag.data).cpu().sum()
            running_results['acc_Task'] += 100. * correct_Task / total_Task
            lossdict['train/Task_subnet'] += loss_Task.item()

            _, predicted_IDH = torch.max(pred_IDH.data, 1)
            total_IDH = label_IDH.size(0)
            correct_IDH = predicted_IDH.eq(label_IDH.data).cpu().sum()
            _, predicted_1p19q = torch.max(pred_1p19q.data, 1)
            total_1p19q = label_1p19q.size(0)
            correct_1p19q = predicted_1p19q.eq(label_1p19q.data).cpu().sum()
            _, predicted_CDKN = torch.max(pred_CDKN.data, 1)
            total_CDKN = label_CDKN.size(0)
            correct_CDKN = predicted_CDKN.eq(label_CDKN.data).cpu().sum()
            _, predicted_His = torch.max(pred_His.data, 1)
            total_His = label_His.size(0)
            correct_His = predicted_His.eq(label_His.data).cpu().sum()

            # _, predicted_Grade = torch.max(pred_Grade.data, 1)
            # total_Grade = label_Grade.size(0)
            # correct_Grade = predicted_Grade.eq(label_Grade.data).cpu().sum()

            total_Diag=total_IDH
            correct_Diag=Diag_process(label,pred_IDH,pred_1p19q,pred_CDKN,pred_His_2class)


            running_results['acc_IDH'] += 100. * correct_IDH / total_IDH
            running_results['acc_1p19q'] += 100. * correct_1p19q / total_1p19q
            if total_CDKN != 0:
                running_results['acc_CDKN'] += 100. * correct_CDKN / total_CDKN
            running_results['acc_His'] += 100. * correct_His / total_His
            # running_results['acc_Grade'] += 100. * correct_Grade/ total_Grade
            running_results['acc_Diag'] += 100. * correct_Diag / total_Diag


            total_it = total_it + 1
            lossdict['train/IDH_subnet'] += loss_IDH.item()
            lossdict['train/1p19q_subnet'] += loss_1p19q.item()
            if isinstance(loss_CDKN, torch.Tensor):
                lossdict['train/CDKN_subnet'] += loss_CDKN.item()
            lossdict['train/Graph_subnet'] += loss_Graph.item()
            lossdict['train/His_subnet'] += loss_His.item()



            train_bar.set_description(
                desc=opt['name'] + ' [%d/%d] I:%.2f |1:%.2f |C:%.2f |H:%.2f |D:%.2f |T:%.2f' % (
                    epoch, alleps,
                    running_results['acc_IDH'] / count,
                    running_results['acc_1p19q'] / count,
                    running_results['acc_CDKN'] / count,
                    running_results['acc_His'] / count,
                    running_results['acc_Diag'] / count,
                    running_results['acc_Task'] / count,
                ))
        lossdict['train/IDH_subnet'] = lossdict['train/IDH_subnet'] / count
        lossdict['train/1p19q_subnet'] = lossdict['train/1p19q_subnet'] / count
        lossdict['train/CDKN_subnet'] = lossdict['train/CDKN_subnet'] / count
        lossdict['train/Graph_subnet'] = lossdict['train/Graph_subnet'] / count
        lossdict['train/Grade_subnet'] = lossdict['train/Grade_subnet'] / count
        lossdict['train/Task_subnet'] = lossdict['train/Task_subnet'] / count
        saver.write_scalars(curep, lossdict)
        saver.write_log(curep, lossdict, 'traininglossLog')



        
        if (curep + 1)>=1:
            print('-------------------------------------Val and Test--------------------------------------')
            if (curep + 1) > (1):
                save_dir = os.path.join(opt['modelDir'], 'Mine_model-%04d.pth' % (curep + 1))
                state = {
                    'init': Mine_model_init.state_dict(),
                    'IDH': Mine_model_IDH.state_dict(),
                    '1p19q': Mine_model_1p19q.state_dict(),
                    'CDKN': Mine_model_CDKN.state_dict(),
                    'Graph': Mine_model_Graph.state_dict(),
                    'His': Mine_model_His.state_dict(),
                    'Cls': Mine_model_Cls.state_dict(),
                    'Task': Mine_model_Task.state_dict(),
                }
                torch.save(state, save_dir)

            print("----------Test-------------")
            list_WSI_IDH,list_WSI_1p19q,list_WSI_CDKN,list_WSI_His_2class,list_WSI_Diag= \
                validation_All(opt, Mine_model_init, Mine_model_IDH, Mine_model_1p19q,Mine_model_CDKN,Mine_model_Graph,Mine_model_His,Mine_model_Cls,Mine_model_Task,testLoader, saver, curep + 1, opt['eva_cm'], gpuID, task = '')
            print('test in epoch: %d/%d, acc_IDH:%.3f,acc_1p19q:%.3f,acc_CDKN:%.3f,acc_His_2class:%.3f, acc_Diag:%.3f' % (
                curep + 1, alleps, list_WSI_IDH[0], list_WSI_1p19q[0], list_WSI_CDKN[0], list_WSI_His_2class[0], list_WSI_Diag[0]))
            test_dict = {'test/acc_IDH': list_WSI_IDH[0], 'test/sen_IDH': list_WSI_IDH[3], 'test/spec_IDH': list_WSI_IDH[4],
                        'test/auc_IDH': list_WSI_IDH[5],'test/f1_IDH': list_WSI_IDH[2], 'test/prec_IDH': list_WSI_IDH[6],
                        'test/acc_1p19q': list_WSI_1p19q[0], 'test/sen_1p19q': list_WSI_1p19q[3], 'test/spec_1p19q': list_WSI_1p19q[4],
                        'test/auc_1p19q': list_WSI_1p19q[5], 'test/f1_1p19q': list_WSI_1p19q[2], 'test/prec_1p19q': list_WSI_1p19q[6],
                        'test/acc_CDKN': list_WSI_CDKN[0], 'test/sen_CDKN': list_WSI_CDKN[3], 'test/spec_CDKN': list_WSI_CDKN[4],
                        'test/auc_CDKN': list_WSI_CDKN[5], 'test/f1_CDKN': list_WSI_CDKN[2], 'test/prec_CDKN': list_WSI_CDKN[6],
                        'test/acc_His_2class': list_WSI_His_2class[0], 'test/sen_His_2class': list_WSI_His_2class[3], 'test/spec_His_2class': list_WSI_His_2class[4],
                        'test/auc_His_2class': list_WSI_His_2class[5], 'test/f1_His_2class': list_WSI_His_2class[2], 'test/prec_His_2class': list_WSI_His_2class[6],
                        'test/acc_Diag': list_WSI_Diag[0], 'test/sen_Diag': list_WSI_Diag[3], 'test/spec_Diag': list_WSI_Diag[4],
                        'test/auc_Diag': list_WSI_Diag[5], 'test/f1_Diag':  list_WSI_Diag[2], 'test/prec_Diag': list_WSI_Diag[6],

                        }
            test_dict_IDH = {'test/acc_IDH': list_WSI_IDH[0], 'test/sen_IDH': list_WSI_IDH[3], 'test/spec_IDH': list_WSI_IDH[4],
                        'test/auc_IDH': list_WSI_IDH[5], 'test/f1_IDH': list_WSI_IDH[2], 'test/prec_IDH': list_WSI_IDH[6],}
            test_dict_1p19q = {'test/acc_1p19q': list_WSI_1p19q[0], 'test/sen_1p19q': list_WSI_1p19q[3],'test/spec_1p19q': list_WSI_1p19q[4],
                        'test/auc_1p19q': list_WSI_1p19q[5], 'test/f1_1p19q': list_WSI_1p19q[2],'test/prec_1p19q': list_WSI_1p19q[6],}
            test_dict_CDKN = {'test/acc_CDKN': list_WSI_CDKN[0], 'test/sen_CDKN': list_WSI_CDKN[3],'test/spec_CDKN': list_WSI_CDKN[4],
                        'test/auc_CDKN': list_WSI_CDKN[5], 'test/f1_CDKN': list_WSI_CDKN[2],'test/prec_CDKN': list_WSI_CDKN[6],}
            test_dict_His_2class = {'test/acc_His_2class': list_WSI_His_2class[0], 'test/sen_His_2class': list_WSI_His_2class[3],'test/spec_His_2class': list_WSI_His_2class[4],
                        'test/auc_His_2class': list_WSI_His_2class[5], 'test/f1_His_2class': list_WSI_His_2class[2],'test/prec_His_2class': list_WSI_His_2class[6],}
            test_dict_Diag = {'test/acc_Diag': list_WSI_Diag[0], 'test/sen_Diag': list_WSI_Diag[3],'test/spec_Diag': list_WSI_Diag[4],
                        'test/auc_Diag': list_WSI_Diag[5], 'test/f1_Diag': list_WSI_Diag[2],'test/prec_Diag': list_WSI_Diag[6], }
            saver.write_scalars(curep + 1, test_dict)
            saver.write_log(curep + 1, test_dict_IDH, 'test_IDH')
            saver.write_log(curep + 1, test_dict_1p19q, 'test_1p19q')
            saver.write_log(curep + 1, test_dict_CDKN, 'test_CDKN')
            saver.write_log(curep + 1, test_dict_His_2class, 'test_His_2class')
            saver.write_log(curep + 1, test_dict_Diag, 'test_Diag')

            print("----------ITest-------------")
            list_WSI_IDH,list_WSI_1p19q,list_WSI_CDKN,list_WSI_His_2class,list_WSI_Diag= \
                validation_All(opt, Mine_model_init, Mine_model_IDH, Mine_model_1p19q,Mine_model_CDKN,Mine_model_Graph,Mine_model_His,Mine_model_Cls,Mine_model_Task,ItestLoader, saver, curep + 1, opt['eva_cm'], gpuID, task = 'Itest')
            print('Itest in epoch: %d/%d, acc_IDH:%.3f,acc_1p19q:%.3f,acc_CDKN:%.3f,acc_His_2class:%.3f, acc_Diag:%.3f' % (
                curep + 1, alleps, list_WSI_IDH[0], list_WSI_1p19q[0], list_WSI_CDKN[0], list_WSI_His_2class[0], list_WSI_Diag[0]))
            Itest_dict = {'Itest/acc_IDH': list_WSI_IDH[0], 'Itest/sen_IDH': list_WSI_IDH[3], 'Itest/spec_IDH': list_WSI_IDH[4],
                        'Itest/auc_IDH': list_WSI_IDH[5],'Itest/f1_IDH': list_WSI_IDH[2], 'Itest/prec_IDH': list_WSI_IDH[6],
                        'Itest/acc_1p19q': list_WSI_1p19q[0], 'Itest/sen_1p19q': list_WSI_1p19q[3], 'Itest/spec_1p19q': list_WSI_1p19q[4],
                        'Itest/auc_1p19q': list_WSI_1p19q[5], 'Itest/f1_1p19q': list_WSI_1p19q[2], 'Itest/prec_1p19q': list_WSI_1p19q[6],
                        'Itest/acc_CDKN': list_WSI_CDKN[0], 'Itest/sen_CDKN': list_WSI_CDKN[3], 'Itest/spec_CDKN': list_WSI_CDKN[4],
                        'Itest/auc_CDKN': list_WSI_CDKN[5], 'Itest/f1_CDKN': list_WSI_CDKN[2], 'Itest/prec_CDKN': list_WSI_CDKN[6],
                        'Itest/acc_His_2class': list_WSI_His_2class[0], 'Itest/sen_His_2class': list_WSI_His_2class[3], 'Itest/spec_His_2class': list_WSI_His_2class[4],
                        'Itest/auc_His_2class': list_WSI_His_2class[5], 'Itest/f1_His_2class': list_WSI_His_2class[2], 'Itest/prec_His_2class': list_WSI_His_2class[6],
                        'Itest/acc_Diag': list_WSI_Diag[0], 'Itest/sen_Diag': list_WSI_Diag[3], 'Itest/spec_Diag': list_WSI_Diag[4],
                        'Itest/auc_Diag': list_WSI_Diag[5], 'Itest/f1_Diag':  list_WSI_Diag[2], 'Itest/prec_Diag': list_WSI_Diag[6],
                        }
            Itest_dict_IDH = {'Itest/acc_IDH': list_WSI_IDH[0], 'Itest/sen_IDH': list_WSI_IDH[3], 'Itest/spec_IDH': list_WSI_IDH[4],
                        'Itest/auc_IDH': list_WSI_IDH[5], 'Itest/f1_IDH': list_WSI_IDH[2], 'Itest/prec_IDH': list_WSI_IDH[6],}
            Itest_dict_1p19q = {'Itest/acc_1p19q': list_WSI_1p19q[0], 'Itest/sen_1p19q': list_WSI_1p19q[3],'Itest/spec_1p19q': list_WSI_1p19q[4],
                        'Itest/auc_1p19q': list_WSI_1p19q[5], 'Itest/f1_1p19q': list_WSI_1p19q[2],'Itest/prec_1p19q': list_WSI_1p19q[6],}
            Itest_dict_CDKN = {'Itest/acc_CDKN': list_WSI_CDKN[0], 'Itest/sen_CDKN': list_WSI_CDKN[3],'Itest/spec_CDKN': list_WSI_CDKN[4],
                        'Itest/auc_CDKN': list_WSI_CDKN[5], 'Itest/f1_CDKN': list_WSI_CDKN[2],'Itest/prec_CDKN': list_WSI_CDKN[6],}
            Itest_dict_His_2class = {'Itest/acc_His_2class': list_WSI_His_2class[0], 'Itest/sen_His_2class': list_WSI_His_2class[3],'Itest/spec_His_2class': list_WSI_His_2class[4],
                        'Itest/auc_His_2class': list_WSI_His_2class[5], 'Itest/f1_His_2class': list_WSI_His_2class[2],'Itest/prec_His_2class': list_WSI_His_2class[6],}
            Itest_dict_Diag = {'Itest/acc_Diag': list_WSI_Diag[0], 'Itest/sen_Diag': list_WSI_Diag[3],'Itest/spec_Diag': list_WSI_Diag[4],
                        'Itest/auc_Diag': list_WSI_Diag[5], 'Itest/f1_Diag': list_WSI_Diag[2],'Itest/prec_Diag': list_WSI_Diag[6], }
            saver.write_scalars(curep + 1, Itest_dict)
            saver.write_log(curep + 1, Itest_dict_IDH, 'Itest_IDH')
            saver.write_log(curep + 1, Itest_dict_1p19q, 'Itest_1p19q')
            saver.write_log(curep + 1, Itest_dict_CDKN, 'Itest_CDKN')
            saver.write_log(curep + 1, Itest_dict_His_2class, 'Itest_His_2class')
            saver.write_log(curep + 1, Itest_dict_Diag, 'Itest_Diag')
def remove_all_file(path):
    if os.path.isdir(path):
        for i in os.listdir(path):
            path_file = os.path.join(path, i)
            os.remove(path_file)
def remove_all_dir(path):
    if os.path.isdir(path):
        for i in os.listdir(path):
            path_file = os.path.join(path, i)
            for j in os.listdir(path_file):
                path_file1 = os.path.join(path_file, j)
                os.remove(path_file1)
            os.rmdir(path_file)





if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--opt', type=str, default='config/mine.yml')
    args = parser.parse_args()
    with open(args.opt) as f:
        opt = yaml.load(f, Loader=SafeLoader)


    sysstr = platform.system()


    setup_seed(opt['seed'])
    if opt['command']=='Train':
        cur_time = time.strftime('%m%d-%H%M', time.localtime())


        opt['name'] = 'noGraph'+'_{}'.format(cur_time)
        opt['logDir'] = os.path.join(opt['logDir'], opt['name'])
        opt['modelDir'] = os.path.join(opt['modelDir'], opt['name'])
        opt['saveDir'] = os.path.join(opt['saveDir'], opt['name'])
        opt['cm_saveDir'] = os.path.join(opt['cm_saveDir'], opt['name'])
        if not os.path.exists(opt['logDir']):
            os.makedirs(opt['logDir'])
        if not os.path.exists(opt['modelDir']):
            os.makedirs(opt['modelDir'])
        if not os.path.exists(opt['saveDir']):
            os.makedirs(opt['saveDir'])
        if not os.path.exists(opt['cm_saveDir']):
            os.makedirs(opt['cm_saveDir'])

        para_log = os.path.join(opt['modelDir'], 'params.yml')
        if os.path.exists(para_log):
            os.remove(para_log)
        with open(para_log, 'w') as f:
            data = yaml.dump(opt, f, sort_keys=False, default_flow_style=False)

        print("\n\n============> begin training <=======")
        train(opt)




    a=1





























