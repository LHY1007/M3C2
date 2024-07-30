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
    opt['gpus'] = [5]
    gpuID = opt['gpus']
    opt['batchSize'] =1
    
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



    ###############  Datasets #######################
    root_init =r'/home/hanyu/LHY/miccai7.22/best_model/Best3_0720-0755-0003.pth'
    ckptdir_init = os.path.join(root_init)
    checkpoint_init = torch.load(ckptdir_init)
    root_IDH =r'/home/hanyu/LHY/miccai7.22/best_model/Best3_0720-0755-0003.pth'
    ckptdir_IDH = os.path.join(root_IDH)
    checkpoint_IDH = torch.load(ckptdir_IDH)
    root_1p19q =r'/home/hanyu/LHY/miccai7.22/best_model/Best3_0720-0755-0003.pth'
    ckptdir_1p19q = os.path.join(root_1p19q)
    checkpoint_1p19q = torch.load(ckptdir_1p19q)
    root_CDKN =r'/home/hanyu/LHY/miccai7.22/best_model/Best3_0720-0755-0003.pth'
    ckptdir_CDKN = os.path.join(root_CDKN)
    checkpoint_CDKN = torch.load(ckptdir_CDKN)
    root_Task =r'/home/hanyu/LHY/miccai7.22/best_model/Best3_0720-0755-0003.pth'
    ckptdir_Task = os.path.join(root_Task)
    checkpoint_Task = torch.load(ckptdir_Task)

    related_params = {k: v for k, v in checkpoint_init['init'].items()}
    Mine_model_init.load_state_dict(related_params)
    related_params = {k: v for k, v in checkpoint_IDH['IDH'].items()}
    Mine_model_IDH.load_state_dict(related_params)
    related_params = {k: v for k, v in checkpoint_1p19q['1p19q'].items()}
    Mine_model_1p19q.load_state_dict(related_params)
    related_params = {k: v for k, v in checkpoint_CDKN['CDKN'].items()}
    Mine_model_CDKN.load_state_dict(related_params)
    related_params = {k: v for k, v in checkpoint_IDH['Graph'].items()}
    Mine_model_Graph.load_state_dict(related_params)
    related_params = {k: v for k, v in checkpoint_IDH['His'].items()}
    Mine_model_His.load_state_dict(related_params)
    related_params = {k: v for k, v in checkpoint_IDH['Cls'].items()}
    Mine_model_Cls.load_state_dict(related_params)
    related_params = {k: v for k, v in checkpoint_Task['Task'].items()}
    Mine_model_Task.load_state_dict(related_params)
    
    Mine_model_init.eval()
    Mine_model_IDH.eval()
    Mine_model_1p19q.eval()
    Mine_model_CDKN.eval()
    Mine_model_Graph.eval()
    Mine_model_His.eval()
    Mine_model_Cls.eval()
    Mine_model_Task.eval()


    for i in range(1):
        testDataset = dataset_mine.Our_Dataset(phase='Test',opt=opt)
        testLoader = DataLoader(testDataset, batch_size=opt['Test_batchSize'],
                                num_workers=opt['nThreads'] if (sysstr == "Linux") else 1, shuffle=False)
        ItestDataset = dataset_mine.Our_Dataset(phase='ITest',opt=opt)
        ItestLoader = DataLoader(ItestDataset, batch_size=opt['Test_batchSize'],
                                num_workers=opt['nThreads'] if (sysstr == "Linux") else 1, shuffle=False)

        last_ep = 0
        saver = Saver(opt)
        alleps = opt['n_ep'] - last_ep
        curep=0
        print('-------------------------------------Val and Test--------------------------------------')
        if (curep + 1) > (2):
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

        # print("----------ITest-------------")
        # list_WSI_IDH,list_WSI_1p19q,list_WSI_CDKN,list_WSI_His_2class,list_WSI_Diag= \
        #     validation_All(opt, Mine_model_init, Mine_model_IDH, Mine_model_1p19q,Mine_model_CDKN,Mine_model_Graph,Mine_model_His,Mine_model_Cls,Mine_model_Task,ItestLoader, saver, curep + 1, opt['eva_cm'], gpuID, task = 'Itest')
        # print('Itest in epoch: %d/%d, acc_IDH:%.3f,acc_1p19q:%.3f,acc_CDKN:%.3f,acc_His_2class:%.3f, acc_Diag:%.3f' % (
        #     curep + 1, alleps, list_WSI_IDH[0], list_WSI_1p19q[0], list_WSI_CDKN[0], list_WSI_His_2class[0], list_WSI_Diag[0]))
        # Itest_dict = {'Itest/acc_IDH': list_WSI_IDH[0], 'Itest/sen_IDH': list_WSI_IDH[3], 'Itest/spec_IDH': list_WSI_IDH[4],
        #             'Itest/auc_IDH': list_WSI_IDH[5],'Itest/f1_IDH': list_WSI_IDH[2], 'Itest/prec_IDH': list_WSI_IDH[6],
        #             'Itest/acc_1p19q': list_WSI_1p19q[0], 'Itest/sen_1p19q': list_WSI_1p19q[3], 'Itest/spec_1p19q': list_WSI_1p19q[4],
        #             'Itest/auc_1p19q': list_WSI_1p19q[5], 'Itest/f1_1p19q': list_WSI_1p19q[2], 'Itest/prec_1p19q': list_WSI_1p19q[6],
        #             'Itest/acc_CDKN': list_WSI_CDKN[0], 'Itest/sen_CDKN': list_WSI_CDKN[3], 'Itest/spec_CDKN': list_WSI_CDKN[4],
        #             'Itest/auc_CDKN': list_WSI_CDKN[5], 'Itest/f1_CDKN': list_WSI_CDKN[2], 'Itest/prec_CDKN': list_WSI_CDKN[6],
        #             'Itest/acc_His_2class': list_WSI_His_2class[0], 'Itest/sen_His_2class': list_WSI_His_2class[3], 'Itest/spec_His_2class': list_WSI_His_2class[4],
        #             'Itest/auc_His_2class': list_WSI_His_2class[5], 'Itest/f1_His_2class': list_WSI_His_2class[2], 'Itest/prec_His_2class': list_WSI_His_2class[6],
        #             'Itest/acc_Diag': list_WSI_Diag[0], 'Itest/sen_Diag': list_WSI_Diag[3], 'Itest/spec_Diag': list_WSI_Diag[4],
        #             'Itest/auc_Diag': list_WSI_Diag[5], 'Itest/f1_Diag':  list_WSI_Diag[2], 'Itest/prec_Diag': list_WSI_Diag[6],
        #             }
        # Itest_dict_IDH = {'Itest/acc_IDH': list_WSI_IDH[0], 'Itest/sen_IDH': list_WSI_IDH[3], 'Itest/spec_IDH': list_WSI_IDH[4],
        #             'Itest/auc_IDH': list_WSI_IDH[5], 'Itest/f1_IDH': list_WSI_IDH[2], 'Itest/prec_IDH': list_WSI_IDH[6],}
        # Itest_dict_1p19q = {'Itest/acc_1p19q': list_WSI_1p19q[0], 'Itest/sen_1p19q': list_WSI_1p19q[3],'Itest/spec_1p19q': list_WSI_1p19q[4],
        #             'Itest/auc_1p19q': list_WSI_1p19q[5], 'Itest/f1_1p19q': list_WSI_1p19q[2],'Itest/prec_1p19q': list_WSI_1p19q[6],}
        # Itest_dict_CDKN = {'Itest/acc_CDKN': list_WSI_CDKN[0], 'Itest/sen_CDKN': list_WSI_CDKN[3],'Itest/spec_CDKN': list_WSI_CDKN[4],
        #             'Itest/auc_CDKN': list_WSI_CDKN[5], 'Itest/f1_CDKN': list_WSI_CDKN[2],'Itest/prec_CDKN': list_WSI_CDKN[6],}
        # Itest_dict_His_2class = {'Itest/acc_His_2class': list_WSI_His_2class[0], 'Itest/sen_His_2class': list_WSI_His_2class[3],'Itest/spec_His_2class': list_WSI_His_2class[4],
        #             'Itest/auc_His_2class': list_WSI_His_2class[5], 'Itest/f1_His_2class': list_WSI_His_2class[2],'Itest/prec_His_2class': list_WSI_His_2class[6],}
        # Itest_dict_Diag = {'Itest/acc_Diag': list_WSI_Diag[0], 'Itest/sen_Diag': list_WSI_Diag[3],'Itest/spec_Diag': list_WSI_Diag[4],
        #             'Itest/auc_Diag': list_WSI_Diag[5], 'Itest/f1_Diag': list_WSI_Diag[2],'Itest/prec_Diag': list_WSI_Diag[6], }
        # saver.write_scalars(curep + 1, Itest_dict)
        # saver.write_log(curep + 1, Itest_dict_IDH, 'Itest_IDH')
        # saver.write_log(curep + 1, Itest_dict_1p19q, 'Itest_1p19q')
        # saver.write_log(curep + 1, Itest_dict_CDKN, 'Itest_CDKN')
        # saver.write_log(curep + 1, Itest_dict_His_2class, 'Itest_His_2class')
        # saver.write_log(curep + 1, Itest_dict_Diag, 'Itest_Diag')

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


        opt['name'] = 'Pretrain'+'_{}'.format(cur_time)
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





























