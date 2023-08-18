# -*- coding: utf-8 -*-
from fl_aggregator_libs import *
from random import Random
import scipy.io

initiate_aggregator_setting()

for i in range(torch.cuda.device_count()):
    try:
        device_id=args.gpu_device
        device = torch.device('cuda:'+str(device_id))
        # torch.cuda.set_device(i)
        logging.info(f'End up with cuda device {torch.rand(1).to(device=device)}')
        break
    except Exception as e:
        assert i != torch.cuda.device_count()-1, 'Can not find a feasible GPU'

entire_train_data = None
sample_size_dic = {}

sampledClientSet = set()

os.environ['MASTER_ADDR'] = args.ps_ip
os.environ['MASTER_PORT'] = args.ps_port
# os.environ['NCCL_SOCKET_IFNAME'] = 'enp0s20u1u5'
# os.environ['NCCL_DEBUG'] = 'INFO'


def initiate_sampler_query(queue, numOfClients):
    global logDir
    # Initiate the clientSampler
    if args.sampler_path is None:
    # if not args.load_model and args.sampler_path is None:
        client_sampler = clientSampler(args.sample_mode, args.score_mode, args=args, filter=args.filter_less, sample_seed=args.sample_seed)
    else:
        # load sampler
        args.sampler_path = os.path.join(args.model_path,'aggregator/clientInfoFile')
        with open(args.sampler_path, 'rb') as loader:
            client_sampler = pickle.load(loader)
        logging.info("====Load sampler successfully\n")

    # load client profiles
    global_client_profile = {}
    if os.path.exists(args.client_path):
        with open(args.client_path, 'rb') as fin:
            # {clientId: [computer, bandwidth]}
            global_client_profile = pickle.load(fin)
            # global_client_profile = dict([(key,global_client_profile[key]) for key in range(1,args.numofclients)])

    collectedClients = 0
    initial_time = time.time()
    clientId = 1
    passed = False
    num_client_profile = max(1, len(global_client_profile))

    # In this simulation, we run data split on each worker, which amplifies the # of datasets
    # Waiting for the data information from clients, or timeout
    if args.enable_obs_client:
        roundDurationList=[]
        roundDurationLocalList=[]
        roundDurationCommList=[]
        computationList=[]
        communicationList=[]
    while collectedClients < numOfClients or (time.time() - initial_time) > 5000:
        if not queue.empty():
            tmp_dict = queue.get()

            # we only need to go over once
            if not passed and args.sampler_path is None:
                rank_src = list(tmp_dict.keys())[0]
                distanceVec = tmp_dict[rank_src][0]
                sizeVec = tmp_dict[rank_src][1]
                for index, dis in enumerate(distanceVec): #  len(distanceVec) = 2187
                    # since the worker rankId starts from 1, we also configure the initial dataId as 1
                    mapped_id = max(1, clientId%num_client_profile)
                    systemProfile = global_client_profile[mapped_id] if mapped_id in global_client_profile else [1.0, 1.0]
                    
                    client_sampler.registerClient(rank_src, clientId, dis, sizeVec[index], speed=systemProfile)
                    if args.sample_mode != "random":
                        client_sampler.registerDuration(clientId,
                                batch_size=args.batch_size, upload_epoch=args.upload_epoch,
                                model_size=args.model_size*args.clock_factor)
                    if args.enable_obs_client:
                        roundDuration,roundDurationLocal,roundDurationComm = client_sampler.getCompletionTime(clientId,
                                batch_size=args.batch_size, upload_epoch=args.upload_epoch,
                                model_size=args.model_size* args.clock_factor)
                        roundDurationList.append(roundDuration)
                        roundDurationLocalList.append(roundDurationLocal)
                        roundDurationCommList.append(roundDurationComm)
                        computationList.append(systemProfile['computation'])
                        communicationList.append(systemProfile['communication'])

                    clientId += 1
                passed = True
            collectedClients += 1
    logging.info("====Info of all feasible clients {}".format(client_sampler.getDataInfo())) #{'total_feasible_clients': 491, 'total_length': 33443}
    # logging.info("feasibleClients {}".format(client_sampler.getAllClients()))
    if args.enable_obs_client:
        scipy.io.savemat(logDir+'/obs_client_time.mat', dict(roundDurationList=roundDurationList,
                        roundDurationLocalList=roundDurationLocalList,
                        roundDurationCommList=roundDurationCommList,
                        computationList=computationList,
                        communicationList=communicationList))
        logging.info("====Save obs_client====")
        stop_signal.put(1)

    client_sampler.getRanking()

    return client_sampler

def init_myprocesses(rank, size, model, queue, param_q, stop_signal, fn, backend):
    global sampledClientSet

    dist.init_process_group(backend, rank=rank, world_size=size)

    # After collecting all data information, then decide the clientId to run
    workerRanks = [int(v) for v in str(args.learners).split('-')]
    clientSampler = initiate_sampler_query(queue, len(workerRanks))

    clientIdsToRun = []
    for wrank in workerRanks:
        nextClientIdToRun = clientSampler.nextClientIdToRun(hostId=wrank)
        clientSampler.leaderOnHost([nextClientIdToRun], wrank) #  in clientSampler: self.clientOnHosts[wrank] = [nextClientIdToRun]
        clientIdsToRun.append([nextClientIdToRun]) 
        sampledClientSet.add(nextClientIdToRun)
    # logging.info("leaderOnHost is done") # check OK
    clientTensor = torch.tensor(clientIdsToRun, dtype=torch.int, device=device)
    dist.broadcast(tensor=clientTensor, src=0)

    # Start the PS service
    fn(model, queue, param_q, stop_signal, clientSampler)

def prune_client_tasks(clientSampler, sampledLeadersTemp, numToRealRun, global_virtual_clock):
    sampledLeadersReal = []
    # 1. remove dummy clients that are not available to the end of training
    for virtualLeader in sampledLeadersTemp:
        roundDuration,roundDurationLocal,roundDurationComm = clientSampler.getCompletionTime(virtualLeader,
                                batch_size=args.batch_size, upload_epoch=args.upload_epoch,
                                model_size=args.model_size * args.clock_factor)
        if clientSampler.isClientActive(virtualLeader, roundDuration + global_virtual_clock):
            sampledLeadersReal.append(virtualLeader)

    # 2. we decide to simulate the wall time and remove 1. stragglers 2. off-line
    completionTimes = []
    completionTimesLocal = []
    completionTimesComm = []
    rewardListRaw = []
    virtual_clock = {}
    for virtualLeader in sampledLeadersReal:
        roundDuration,roundDurationLocal,roundDurationComm = clientSampler.getCompletionTime(virtualLeader,batch_size=args.batch_size, 
                                                            upload_epoch=args.upload_epoch,model_size=args.model_size * args.clock_factor)
        completionTimes.append(roundDuration)
        completionTimesLocal.append(roundDurationLocal)
        completionTimesComm.append(roundDurationComm)
        feedback=clientSampler.getClientGradient(virtualLeader)
        rewardListRaw.append(feedback['gradient'])
        virtual_clock[virtualLeader] = roundDuration
    logging.info("completionTimes: {}".format(len(completionTimes)))

    # 3. get the top-k completions
    sortedWorkersByCompletion = sorted(range(len(completionTimes)), key=lambda k:completionTimes[k])
    top_k_index = sortedWorkersByCompletion[:numToRealRun]
    clients_to_run = [sampledLeadersReal[k] for k in top_k_index]

    ## TODO: return the adaptive local epoch
    dummy_clients = [sampledLeadersReal[k] for k in sortedWorkersByCompletion[numToRealRun:]]

    prt_Times = []
    for i in top_k_index:
        prt_Times.append(completionTimes[i])

    round_duration = completionTimes[top_k_index[-1]]
    logging.info("top_k round duration: {}".format(round_duration))

    rewardList=[rewardListRaw[k] for k in top_k_index]
    rewardListSorted=sorted(rewardList,reverse=True)
    rewardListRanking=[rewardListSorted.index(rewardList[k]) for k in range(len(rewardList))]

    if args.enable_dropout:
        increment_factor=(args.dropout_high-args.dropout_low)/args.total_worker
        clients_to_run_dropout_ratio = [args.dropout_low+k*increment_factor for k in rewardListRanking]

        for k_index,k in enumerate(top_k_index):
            completionTimes[k]=completionTimesLocal[k]+(1-clients_to_run_dropout_ratio[k_index])*completionTimesComm[k]
        round_duration = max([completionTimes[k] for k in top_k_index])
        logging.info("round duration after drop out: {}".format(round_duration))
    else:
        clients_to_run_dropout_ratio=[0 for k in top_k_index]

    if args.enable_adapt_local_epoch:
        if args.sample_mode == "heaps":
            clients_to_run_local_epoch_ratio = [min(20,args.adaptive_epoch_beta*math.floor((round_duration-completionTimes[k])/(completionTimesLocal[k]/args.upload_epoch))/args.upload_epoch)+1 for k in top_k_index]
        else:
            clients_to_run_local_epoch_ratio = [min(10,args.adaptive_epoch_beta*math.floor((round_duration-completionTimes[k])/(completionTimesLocal[k]/args.upload_epoch))/args.upload_epoch)+1 for k in top_k_index]
    else:
        clients_to_run_local_epoch_ratio=[1 for k in top_k_index]

    if args.enable_obs_local_epoch:
        scipy.io.savemat(logDir+'/obs_local_epoch_time.mat', dict(completionTimes=[completionTimes[k] for k in sortedWorkersByCompletion],completionTimesLocal=[completionTimesLocal[k] for k in sortedWorkersByCompletion],completionTimesComm=[completionTimesComm[k] for k in sortedWorkersByCompletion],rewardListRaw=[rewardListRaw[k] for k in sortedWorkersByCompletion]))
        
    return clients_to_run, dummy_clients, virtual_clock, round_duration, clients_to_run_local_epoch_ratio, clients_to_run_dropout_ratio

def run(model, queue, param_q, stop_signal, clientSampler):
    global logDir, sampledClientSet

    logging.info("====PS: get in run()")

    model = model.to(device=device)

    # with open(args.model_path+'/model.pth.tar', 'wb') as fout:
    #     pickle.dump(model, fout)

    #if not args.load_model:
    for name, param in model.named_parameters():
        dist.broadcast(tensor=param.data.to(device=device), src=0)
        #logging.info(f"====Model parameters name: {name}")

    workers = [int(v) for v in str(args.learners).split('-')]

    epoch_train_loss = 0
    data_size_epoch = 0   # len(train_data), one epoch
    epoch_count = 1
    global_virtual_clock = 0.
    round_duration = 0.

    staleness = 0
    learner_staleness = {l: 0 for l in workers}
    learner_local_step = {l: 0 for l in workers}
    learner_cache_step = {l: 0 for l in workers}
    pendingWorkers = {}
    test_results = {}
    virtualClientClock = {}
    exploredPendingWorkers = []
    avgUtilLastEpoch = 0.
    avgGradientUtilLastEpoch = 0.

    s_time = time.time()
    epoch_time = s_time

    global_update = 0
    received_updates = 0

    clientsLastEpoch = []
    sumDeltaWeights = []
    clientWeightsCache = {}
    last_sampled_leaders = None
    last_sampled_members = None
    last_model_parameters = [torch.clone(p.data) for p in model.parameters()]

    # random component to generate noise
    median_reward = 1.

    gradient_controller = None
    # initiate yogi if necessary
    if args.gradient_policy == 'yogi':
        gradient_controller = YoGi(eta=args.yogi_eta, tau=args.yogi_tau, beta=args.yogi_beta, beta2=args.yogi_beta2)

    clientInfoFile = os.path.join(logDir, 'clientInfoFile')
    # dump the client info
    with open(clientInfoFile, 'wb') as fout:
        # pickle.dump(clientSampler.getClientsInfo(), fout)
        pickle.dump(clientSampler, fout)
    if args.load_model:
        training_history_path = os.path.join(args.model_path,'aggregator/training_perf')
        with open(training_history_path, 'rb') as fin:
            training_history = pickle.load(fin)
        load_perf_epoch_retrieved=list(training_history['perf'].keys())
        load_perf_epoch=load_perf_epoch_retrieved[-1]
        load_perf_clock=training_history['perf'][load_perf_epoch]['clock']
    else:
        training_history = {'data_set': args.data_set,
                        'model': args.model,
                        'sample_mode': args.sample_mode,
                        'gradient_policy': args.gradient_policy,
                        'task': args.task,
                        'perf': collections.OrderedDict()}

        load_perf_clock=0
        load_perf_epoch=0

    while True:
        if not queue.empty():
            try:
                handle_start = time.time()
                tmp_dict = queue.get()
                rank_src = list(tmp_dict.keys())[0]

                [iteration_loss, trained_size, isWorkerEnd, clientIds, speed, testRes, virtualClock] = \
                [tmp_dict[rank_src][i] for i in range(1, len(tmp_dict[rank_src]))]
                #clientSampler.registerSpeed(rank_src, clientId, speed)
                if isWorkerEnd:
                    logging.info("====Worker {} has completed all its data computation!".format(rank_src))
                    learner_staleness.pop(rank_src)
                    if (len(learner_staleness) == 0):
                        stop_signal.put(1)
                        break
                    continue
                learner_local_step[rank_src] += 1

                handlerStart = time.time()
                delta_wss = tmp_dict[rank_src][0]
                clientsLastEpoch += clientIds
                ratioSample = 0

                logging.info("====Start to merge models")
                if args.enable_obs_local_epoch and epoch_count >1:
                    gradient_l2_norm_list=[]
                    gradientUtilityList=[]
                if not args.test_only or epoch_count == 1:
                    for i, clientId in enumerate(clientIds):
                        gradients = None
                        ranSamples = float(speed[i].split('_')[1])

                        data_size_epoch += trained_size[i]

                        # fraction of total samples on this specific node
                        ratioSample = clientSampler.getSampleRatio(clientId, rank_src, args.is_even_avg) # ratioSample:0.14285714285714285
                        #logging.info("ratioSample:{}".format(ratioSample)) # check OK
                        delta_ws = delta_wss[i]
                        #clientWeightsCache[clientId] = [torch.from_numpy(x).to(device=device) for x in delta_ws]
                        #TODO:ADD LOSS AVERAGELY
                        epoch_train_loss += ratioSample * iteration_loss[i]
                        isSelected = True if clientId in sampledClientSet else False

                        gradient_l2_norm = 0
                        
                        # apply the update into the global model if the client is involved
                        for idx, param in enumerate(model.parameters()):
                            model_weight = torch.from_numpy(delta_ws[idx]).to(device=device)

                            # model_weight is the delta of last model
                            if isSelected:
                                # the first received client
                                if received_updates == 0:
                                    sumDeltaWeights.append(model_weight * ratioSample)
                                else:
                                    sumDeltaWeights[idx] += model_weight * ratioSample
                            if args.sample_mode == "oort":
                                gradient_l2_norm += ((model_weight-last_model_parameters[idx]).norm(2)**2).item()

                        # bias term for global speed
                        virtual_c = virtualClientClock[clientId] if clientId in virtualClientClock else 1.
                        clientUtility = 1.
                        size_of_sample_bin = 1.

                        if args.capacity_bin == True:
                            if not args.enable_adapt_local_epoch:
                                size_of_sample_bin = min(clientSampler.getClient(clientId).size, args.upload_epoch*args.batch_size) #size_of_sample_bin:74
                            else:
                                size_of_sample_bin = min(clientSampler.getClient(clientId).size, trained_size[i])
                        #logging.info("size_of_sample_bin:{}".format(size_of_sample_bin)) # check OK
                        # register the score
                        clientUtility = math.sqrt(iteration_loss[i]) * size_of_sample_bin
                        gradientUtility = math.sqrt(gradient_l2_norm) * size_of_sample_bin/100
                        if args.enable_obs_local_epoch and epoch_count >1:
                            gradient_l2_norm_list.append(gradient_l2_norm)
                            gradientUtilityList.append(gradientUtility)
                        # add noise to the utility
                        if args.noise_factor > 0:
                            noise = np.random.normal(0, args.noise_factor * median_reward, 1)[0]
                            clientUtility += noise
                            clientUtility = max(1e-2, clientUtility)

                        clientSampler.registerScore(clientId, clientUtility, gradientUtility,auxi=math.sqrt(iteration_loss[i]),time_stamp=epoch_count, duration=virtual_c)

                        if isSelected:
                            received_updates += 1

                        avgUtilLastEpoch += ratioSample * clientUtility
                        avgGradientUtilLastEpoch+=ratioSample *gradientUtility

                logging.info("====Done handling rank {}, with ratio {}, now collected {} clients".format(rank_src, ratioSample, received_updates))
                if args.enable_obs_local_epoch and epoch_count >1:
                    scipy.io.savemat(logDir+'/obs_local_epoch_gradient.mat', dict(gradient_l2_norm_list=gradient_l2_norm_list,
                        gradientUtilityList=gradientUtilityList))
                    logging.info("====Save obs_local_epoch====")
                    stop_signal.put(1)
                # aggregate the test results
                updateEpoch = testRes[-1]
                if updateEpoch not in test_results:
                    # [top_1, top_5, loss, total_size, # of collected ranks]
                    test_results[updateEpoch] = [0., 0., 0., 0., 0]

                if updateEpoch != -1:
                    for idx, c in enumerate(testRes[:-1]): # collect all clients from the first gpu to the last gpu
                        test_results[updateEpoch][idx] += c

                    test_results[updateEpoch][-1] += 1
                    # have collected all ranks
                    if test_results[updateEpoch][-1] == len(workers): 
                        top_1_str = 'top_1: '
                        top_5_str = 'top_5: '
                        try:
                            logging.info("====After aggregation in epoch: {}, virtual_clock: {}, {}: {} % ({}), {}: {} % ({}), test loss: {}, test len: {}"
                                    .format(updateEpoch+load_perf_epoch, global_virtual_clock+load_perf_clock, top_1_str, round(test_results[updateEpoch][0]/test_results[updateEpoch][3]*100.0, 4),
                                    test_results[updateEpoch][0], top_5_str, round(test_results[updateEpoch][1]/test_results[updateEpoch][3]*100.0, 4),
                                    test_results[updateEpoch][1], test_results[updateEpoch][2]/test_results[updateEpoch][3], test_results[updateEpoch][3]))
                            if not args.load_model or epoch_count>2:
                                training_history['perf'][updateEpoch+load_perf_epoch] = {'round': updateEpoch+load_perf_epoch, 'clock': global_virtual_clock+load_perf_clock,
                                    top_1_str: round(test_results[updateEpoch][0]/test_results[updateEpoch][3]*100.0, 4),
                                    top_5_str: round(test_results[updateEpoch][1]/test_results[updateEpoch][3]*100.0, 4),
                                    'loss': test_results[updateEpoch][2]/test_results[updateEpoch][3],
                                    }

                                with open(os.path.join(logDir, 'training_perf'), 'wb') as fout:
                                    pickle.dump(training_history, fout)

                        except Exception as e:
                            logging.info(f"====Error {e}")

                handlerDur = time.time() - handlerStart
                global_update += 1

                # get the current minimum local staleness_sum_epoch
                currentMinStep = min([learner_local_step[key] for key in learner_local_step.keys()])

                staleness += 1
                learner_staleness[rank_src] = staleness

                # if the worker is within the staleness, then continue w/ local cache and do nothing
                # Otherwise, block it
                if learner_local_step[rank_src] >= args.stale_threshold + currentMinStep:
                    pendingWorkers[rank_src] = learner_local_step[rank_src]
                    # lock the worker
                    logging.info("Lock worker " + str(rank_src) + " with localStep " + str(pendingWorkers[rank_src]) +
                                            " , while globalStep is " + str(currentMinStep) + "\n")

                # if the local cache is too stale, then update it
                elif learner_cache_step[rank_src] < learner_local_step[rank_src] - args.stale_threshold:
                    pendingWorkers[rank_src] = learner_local_step[rank_src]

                # release all pending requests, if the staleness does not exceed the staleness threshold in SSP
                handle_dur = time.time() - handle_start

                workersToSend = []

                for pworker in pendingWorkers.keys():
                    # check its staleness
                    if pendingWorkers[pworker] <= args.stale_threshold + currentMinStep:
                        # start to send param, to avoid synchronization problem, first create a copy here?
                        workersToSend.append(pworker)

                del delta_wss, tmp_dict

                if len(workersToSend) > 0:          
                    # assign avg reward to explored, but not ran workers
                    for clientId in exploredPendingWorkers:
                        clientSampler.registerScore(clientId, avgUtilLastEpoch,avgGradientUtilLastEpoch,
                            time_stamp=epoch_count, duration=virtualClientClock[clientId], success=False)

                    workersToSend = sorted(workersToSend)
                    logging.info("====Epoch {} completes {} clients with loss {}".format(epoch_count, len(clientsLastEpoch), epoch_train_loss)) 
                    #, sampled rewards are: \n {} \n========== , {x:clientSampler.getScore(0, x) for x in sorted(clientsLastEpoch)} 
                    epoch_count += 1
                    avgUtilLastEpoch = 0.
                    avgGradientUtilLastEpoch = 0.

                    epoch_train_loss = 0.
                    clientsLastEpoch = []
                    send_start = time.time()

                    # resampling the clients if necessary
                    if epoch_count % args.resampling_interval == 0 or epoch_count == 2:
                        logging.info("====Start to sample for epoch {}, global virtualClock: {}, round_duration: {}"
                                        .format(epoch_count, global_virtual_clock, round_duration))

                        numToSample = int(args.total_worker * args.overcommit) # 130

                        sampledTempLeaders = []
                        sampledTempMembers = []
                        
                        if args.fixed_clients:
                            sampledTempLeaders = last_sampled_leaders
                            if args.sample_mode == "heaps":
                                sampledTempMembers = last_sampled_members
                        else:
                            if args.sample_mode == "oort" or args.sample_mode == "heaps":
                                sampledTempLeaders = sorted(clientSampler.resampleClients(numToSample, cur_time=epoch_count))
                            else:
                                feasibleClients = clientSampler.getFeasibleClients(cur_time=epoch_count)
                                poss_leader=[1 for key in feasibleClients]
                                poss_leader=np.asarray(poss_leader).astype('float64') / np.sum(poss_leader)
                                sampledTempLeaders = list(np.random.choice(feasibleClients, args.total_worker, p=poss_leader, replace=True))
                        if sampledTempLeaders:
                            last_sampled_leaders = sampledTempLeaders
                        else:
                            sampledTempLeaders = last_sampled_leaders

                        if args.sample_mode == "heaps":
                            # remove dummy clients in leaders that we are not going to run
                            leadersToRun, exploredPendingWorkers, virtualClientClock, round_duration, local_epoch_ratio, dropout_ratio = prune_client_tasks(clientSampler, sampledTempLeaders, int(args.total_worker*args.leaderFactor), global_virtual_clock)
                        else:
                            leadersToRun, exploredPendingWorkers, virtualClientClock, round_duration, local_epoch_ratio, dropout_ratio = prune_client_tasks(clientSampler, sampledTempLeaders, args.total_worker, global_virtual_clock)

                        sampledTempMembers = []
                        membersToRun = []
                        Migration_Matrix = []
                        if args.sample_mode == "heaps":
                            startMemberTime = time.time()
                            sampledTempMembers = clientSampler.resampleMembers(round_duration, leadersToRun, cur_time=epoch_count)
                            # 1. remove off-line (non-active) members, remove dummy clients that are not available to the end of training
                            membersToRun = []
                            for virtualMember in sampledTempMembers:
                                roundDuration,roundDurationLocal,_ = clientSampler.getCompletionTime(virtualMember, batch_size=args.batch_size, 
                                                        upload_epoch=args.upload_epoch, model_size=args.model_size * args.clock_factor)
                                if clientSampler.isClientActive(virtualMember, roundDuration + global_virtual_clock):
                                    membersToRun.append(virtualMember)
                            logging.info("leaders:{}, members:{}".format(len(leadersToRun), len(membersToRun)))
                            # 2. generate migration matrix
                            Migration_Matrix, Commu_TimeMatrix = clientSampler.generateMigrationMatrix(leadersToRun, membersToRun)
                            # logging.info("Migration_Matrix: {}".format(Migration_Matrix)) 
                            # 3. generate local iters for each member
                            member_local_ratio = {}
                            for leader in Migration_Matrix:
                                if Migration_Matrix[leader]:
                                    _,_,leader_commu = clientSampler.getCompletionTime(leader, batch_size=args.batch_size, 
                                                        upload_epoch=args.upload_epoch, model_size=args.model_size * args.clock_factor)
                                    logging.info("leader {} has members: {}, commu time of leader to server:{}".format(leader, Migration_Matrix[leader], leader_commu))
                                    for idx,mm in enumerate(Migration_Matrix[leader]):
                                        _,compuLocal,commuLocal = clientSampler.getCompletionTime(mm, batch_size=args.batch_size,
                                                            upload_epoch=args.upload_epoch,model_size=args.model_size * args.clock_factor)
                                        logging.info("commu time of member to server:{}, to leader {}".format(commuLocal, Commu_TimeMatrix[leader][idx]))
                                        if args.enable_adapt_local_epoch:
                                            member_local_ratio[mm] = min(20,args.adaptive_epoch_beta*math.floor((max(round_duration-leader_commu-Commu_TimeMatrix[leader][idx], 0))/(compuLocal/args.upload_epoch))/args.upload_epoch)+1
                                        else:
                                            member_local_ratio[mm] = 1
                            DurMemberMatrix = time.time() - startMemberTime
                            logging.info("DurMemberMatrix: {}, completionTimes for members: {}".format(DurMemberMatrix, len(member_local_ratio)))
                        last_sampled_members = sampledTempMembers
                        sampledClientSet = set(leadersToRun + membersToRun) if args.sample_mode == "heaps" else set(leadersToRun)
                        # logging.info("====Try to resample clients, final takes: \n {} \n {}".format(list(sampledClientSet)))

                        if args.sample_mode == "heaps":
                            allocateClientToWorker = {}
                            allocateClientLocalEpochToWorker = {}
                            allocateClientDropoutRatioToWorker = {}
                            allocateClientDict = {rank:0 for rank in workers}
                            # for those device lakes < # of clients, we use round-bin for load balance
                            for rank in workers:
                                allocateClientToWorker[rank] = []
                                allocateClientLocalEpochToWorker[rank]=[]
                                allocateClientDropoutRatioToWorker[rank]=[]
                            for idc, c in enumerate(leadersToRun):
                                clientDataSize = clientSampler.getClientSize(c)
                                numOfBatches = int(math.ceil(clientDataSize/args.batch_size))
                                if numOfBatches > args.upload_epoch:
                                    workerId = workers[(c-1)%len(workers)]
                                else:
                                    # pick the one w/ the least load
                                    workerId = sorted(allocateClientDict, key=allocateClientDict.get)[0]
                                # if workerId not in allocateClientToWorker:
                                #     allocateClientToWorker[workerId] = []
                                #     allocateClientLocalEpochToWorker[workerId]=[]
                                #     allocateClientDropoutRatioToWorker[workerId]=[]
                                allocateClientToWorker[workerId].append(c)
                                allocateClientLocalEpochToWorker[workerId].append(local_epoch_ratio[idc])
                                allocateClientDropoutRatioToWorker[workerId].append(dropout_ratio[idc])
                                allocateClientDict[workerId] = allocateClientDict[workerId] + 1
                            # logging.info("allocateClientToWorker.keys() are: {}".format(allocateClientToWorker.keys()))
                        else: 
                            allocateClientToWorker = {}
                            allocateClientLocalEpochToWorker = {}
                            allocateClientDropoutRatioToWorker = {}
                            allocateClientDict = {rank:0 for rank in workers}
                            # for those device lakes < # of clients, we use round-bin for load balance
                            for idc, c in enumerate(leadersToRun):
                                clientDataSize = clientSampler.getClientSize(c)
                                numOfBatches = int(math.ceil(clientDataSize/args.batch_size))
                                if numOfBatches > args.upload_epoch:
                                    workerId = workers[(c-1)%len(workers)]
                                else:
                                    # pick the one w/ the least load
                                    workerId = sorted(allocateClientDict, key=allocateClientDict.get)[0]
                                if workerId not in allocateClientToWorker:
                                    allocateClientToWorker[workerId] = []
                                    allocateClientLocalEpochToWorker[workerId]=[]
                                    allocateClientDropoutRatioToWorker[workerId]=[]
                                allocateClientToWorker[workerId].append(c)
                                allocateClientLocalEpochToWorker[workerId].append(local_epoch_ratio[idc])
                                allocateClientDropoutRatioToWorker[workerId].append(dropout_ratio[idc])
                                allocateClientDict[workerId] = allocateClientDict[workerId] + 1
                            logging.info("allocateClientToWorker.keys() are: {}".format(allocateClientToWorker.keys()))

                        for w in allocateClientToWorker.keys():
                            clientSampler.leaderOnHost(allocateClientToWorker[w], w)
                            clientSampler.clientLocalEpochOnHost(allocateClientLocalEpochToWorker[w], w)
                            clientSampler.clientDropoutratioOnHost(allocateClientDropoutRatioToWorker[w], w)
                        # logging.info("leaders are allocated") # checked OK
                        # allocate members to host by the matrix
                        if args.sample_mode == "heaps" and epoch_count > 2:
                            allocateMemberToWorker = {}
                            allocateMemberLocalEpochToWorker = {}
                            for worker in workers:
                                allocateMemberToWorker[worker] = []
                                allocateMemberLocalEpochToWorker[worker] = []
                                leaders_on_worker = clientSampler.getCurrentLeaderIds(worker)
                                # logging.info("leaders on worker {} are: {}".format(worker, leaders_on_worker)) # checked OK
                                logging.info("allocateLeaderLocalEpochToWorker:{} for worker:{}".format(allocateClientLocalEpochToWorker[worker],worker)) # checked OK
                                for clt in leaders_on_worker:
                                    if Migration_Matrix[clt]:
                                        allocateMemberToWorker[worker] += Migration_Matrix[clt]
                                        allocateMemberLocalEpochToWorker[worker] += [member_local_ratio[k] for k in Migration_Matrix[clt]]
                                clientSampler.memberOnHost(allocateMemberToWorker[worker], worker)
                                clientSampler.memberLocalEpochOnHost(allocateMemberLocalEpochToWorker[worker], worker)
                                members_on_worker = clientSampler.getCurrentMemberIds(worker)
                                logging.info("members on worker {} are: {}".format(worker, members_on_worker)) 
                                logging.info("allocateMemberLocalEpochToWorker:{} for worker:{}".format(allocateMemberLocalEpochToWorker[worker],worker)) # checked OK
                                
                    clientIdsToRun = [currentMinStep]
                    clientsList = []
                    clientsListLocalEpoch = []
                    clientsListDropoutRatio = []
                    endIdx = 0
                    for worker in workers:
                        learner_cache_step[worker] = currentMinStep
                        if args.sample_mode == "heaps":
                            endIdx += clientSampler.getLeaderLenOnHost(worker)
                            # logging.info("leaderLen {}".format(endIdx)) # checked OK
                            if epoch_count > 2:
                                endIdx += clientSampler.getMemberLenOnHost(worker)
                            clientIdsToRun.append(endIdx)
                            # logging.info("client Len {}".format(endIdx)) # checked OK
                            clientsList += clientSampler.getCurrentLeaderIds(worker)
                            if epoch_count > 2:
                                clientsList += clientSampler.getCurrentMemberIds(worker)
                            clientsListLocalEpoch += clientSampler.getCurrentClientLocalEpoch(worker) 
                            if epoch_count > 2:
                                clientsListLocalEpoch += clientSampler.getCurrentMemberLocalEpoch(worker)
                        else:
                            endIdx += clientSampler.getLeaderLenOnHost(worker)
                            clientIdsToRun.append(endIdx)
                            clientsList += clientSampler.getCurrentLeaderIds(worker)
                            clientsListLocalEpoch += clientSampler.getCurrentClientLocalEpoch(worker)
                            clientsListDropoutRatio += clientSampler.getCurrentClientDropoutRatio(worker)
                        # remove from the pending workers
                        del pendingWorkers[worker]
                    
                    # transformation of gradients if necessary
                    if gradient_controller is not None:
                        sumDeltaWeights = gradient_controller.update(sumDeltaWeights)
                    
                    # update the clientSampler and model
                    with open(clientInfoFile, 'wb') as fout:
                        pickle.dump(clientSampler, fout)
                    for idx, param in enumerate(model.parameters()):
                        if not args.test_only:  
                            if (not args.load_model or epoch_count>2):
                                param.data += sumDeltaWeights[idx]
                            dist.broadcast(tensor=(param.data.to(device=device)), src=0)
                    # dist.broadcast(tensor, src, group): Copies tensor from src to all other processes.
                    dist.broadcast(tensor=torch.tensor(clientIdsToRun, dtype=torch.int).to(device=device), src=0) # one worker or gpu runs how many leader clients, size = workers or gpus
                    dist.broadcast(tensor=torch.tensor(clientsList, dtype=torch.int).to(device=device), src=0) # list of all leader clients
                    # logging.info("clientsList {} are broadcast".format(len(clientsList)))
                    dist.broadcast(tensor=torch.tensor(clientsListLocalEpoch, dtype=torch.float).to(device=device), src=0)
                    # logging.info("local epochs {} are broadcast".format(len(clientsListLocalEpoch)))
                    if args.sample_mode == "oort":
                        dist.broadcast(tensor=torch.tensor(clientsListDropoutRatio, dtype=torch.float).to(device=device), src=0)
                    
                    last_model_parameters = [torch.clone(p.data) for p in model.parameters()]
                    if global_update % args.display_step == 0:
                        logging.info("Handle Wight {} | Send {}".format(handle_dur, time.time() - send_start))

                    # update the virtual clock
                    global_virtual_clock += round_duration
                    received_updates = 0

                    sumDeltaWeights = []
                    clientWeightsCache = {}

                    if args.noise_factor > 0:
                        median_reward = clientSampler.get_median_reward()
                        logging.info('For epoch: {}, median_reward: {}, dev: {}'
                                        .format(epoch_count, median_reward, median_reward*args.noise_factor))

                    gc.collect()
                
                # logging.info("ready to next round") # checked OK
                # The training stop
                if(epoch_count >= args.epochs):
                    stop_signal.put(1)
                    logging.info('Epoch is done: {}'.format(epoch_count))
                    break

            except Exception as e:
                exc_type, exc_obj, exc_tb = sys.exc_info()
                fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                print("====Error: " + str(e) + '\n')
                logging.info("====Error: {}, {}, {}, {}".format(e, exc_type, fname, exc_tb.tb_lineno))

        e_time = time.time()
        if (e_time - s_time) >= float(args.timeout):
            stop_signal.put(1)
            print('Time up: {}, Stop Now!'.format(e_time - s_time))
            break

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True

# communication channel for client information
def initiate_channel():

    queue = Queue()
    param = Queue()
    stop_or_not = Queue()

    BaseManager.register('get_queue', callable=lambda: queue)
    BaseManager.register('get_param', callable=lambda: param)
    BaseManager.register('get_stop_signal', callable=lambda: stop_or_not)
    manager = BaseManager(address=(args.ps_ip, args.manager_port), authkey=b'queue')

    return manager

if __name__ == "__main__":

    # Control the global random
    setup_seed(args.this_rank)

    manager = initiate_channel()
    manager.start()

    q = manager.get_queue()  # queue for parameter_server signal process
    param_q = manager.get_param()  # init
    stop_signal = manager.get_stop_signal()  # stop

    logging.info("====Start to initialize dataset")

    model, train_dataset, test_dataset = init_dataset()

    world_size = len(str(args.learners).split('-')) + 1
    this_rank = args.this_rank

    init_myprocesses(this_rank, world_size, model,
                    q, param_q, stop_signal, run, args.backend
                )

    manager.shutdown()