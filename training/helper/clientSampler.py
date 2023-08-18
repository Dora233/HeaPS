from helper.client import Client
import math
from random import Random
import pickle
import logging

import sys
sys.path.insert(0, '../../oort')
from oort import create_training_selector
sys.path.remove('../../oort')
sys.path.insert(0, '../../heaps')
from heaps import build_training_selector
sys.path.remove('../../heaps')

class clientSampler(object):

    def __init__(self, mode, score, args, filter=0, sample_seed=233):
        self.Clients = {}
        self.leaderOnHosts = {}
        self.memberOnHosts = {}
        # self.correspLeaderofMemberOnHosts = {}
        self.clientLocalEpochOnHosts = {}
        self.memberLocalEpochOnHosts = {}
        self.clientDropoutRatioOnHosts = {}
        self.mode = mode
        self.score = score
        self.filter_less = args.filter_less
        self.filter_more = args.filter_more

        if self.mode == 'oort':
            self.ucbSampler = create_training_selector(args=args)
        elif self.mode == 'heaps':
            self.ucbSampler = build_training_selector(args=args)
        else:
            self.ucbSampler = None
        self.feasibleClients = []
        self.rng = Random()
        self.rng.seed(sample_seed)
        self.count = 0
        self.feasible_samples = 0
        self.user_trace = None
        self.args = args

        if args.user_trace is not None:
            with open(args.user_trace, 'rb') as fin:    
                self.user_trace = pickle.load(fin)
        # logging.info("initialization of clientSampler is done") # check OK
        
        self.commuTime_Clients = {}
        self.compuTime_Clients = {}

    def getRanking(self):
        for client in self.feasibleClients:
            _,client_compu,client_commu = self.getCompletionTime(client, batch_size=self.args.batch_size, 
                                upload_epoch=self.args.upload_epoch, model_size=self.args.model_size * self.args.clock_factor)
            self.commuTime_Clients[client] = client_commu
            self.compuTime_Clients[client] = client_compu
        if self.mode == 'heaps':
            self.ucbSampler.generateRankList(self.commuTime_Clients, self.compuTime_Clients)
        
    def registerClient(self, hostId, clientId, dis, size, speed=[1.0, 1.0], duration=1, num_client = 0, compu_rank=0.0, commu_rank=0.0, commuclient_rank=0.0):

        uniqueId = self.getUniqueId(hostId, clientId)
        user_trace = None if self.user_trace is None else self.user_trace[max(1, clientId%len(self.user_trace))]

        self.Clients[uniqueId] = Client(hostId, clientId, dis, size, speed, user_trace)

        # remove clients
        if size >= self.filter_less and size <= self.filter_more: # 32, 1e5
            self.feasibleClients.append(clientId)
            self.feasible_samples += size

            # feedbacks = {'reward':min(size, self.args.upload_epoch*self.args.batch_size),'duration':duration}
            feedbacks = {'reward':size,'duration':duration,'gradient':size,}
            if self.mode != "random":
                self.ucbSampler.register_client(clientId, feedbacks=feedbacks)

    def getAllClients(self):
        return self.feasibleClients

    def getAllClientsLength(self):
        return len(self.feasibleClients)

    def getClient(self, clientId):
        return self.Clients[self.getUniqueId(0, clientId)]

    def registerDuration(self, clientId, batch_size, upload_epoch, model_size):
        if self.mode == "oort":
            roundDuration,roundDurationLocal,roundDurationComm = self.Clients[self.getUniqueId(0, clientId)].getCompletionTime(
                    batch_size=batch_size, upload_epoch=upload_epoch, model_size=model_size
            )
            self.ucbSampler.update_duration(clientId, roundDuration)
        elif self.mode == "heaps":
            roundDuration,roundDurationLocal,roundDurationComm = self.Clients[self.getUniqueId(0, clientId)].getCompletionTime(
                    batch_size=batch_size, upload_epoch=upload_epoch, model_size=model_size
            )
            self.ucbSampler.update_duration(clientId, roundDuration)

    def getCompletionTime(self, clientId, batch_size, upload_epoch, model_size):
        return self.Clients[self.getUniqueId(0, clientId)].getCompletionTime(
                batch_size=batch_size, upload_epoch=upload_epoch, model_size=model_size
            )

    def registerSpeed(self, hostId, clientId, speed):
        uniqueId = self.getUniqueId(hostId, clientId)
        self.Clients[uniqueId].speed = speed

    def registerScore(self, clientId, reward, gradient,auxi=1.0, time_stamp=0, duration=1., success=True):
        # currently, we only use distance as reward
        if self.mode == "oort":
            feedbacks = {
                'reward': reward,
                'gradient': gradient,
                'duration': duration,
                'status': True,
                'time_stamp': time_stamp
            }
            self.ucbSampler.update_client_util(clientId, feedbacks=feedbacks)
        elif self.mode == "heaps":
            feedbacks = {
                'reward': reward,
                'gradient': gradient,
                'duration': duration,
                'status': True,
                'time_stamp': time_stamp
            }
            self.ucbSampler.update_client_util(clientId, feedbacks=feedbacks)
        # logging.info("registerScore is done") # check OK
    # def registerClientScore(self, clientId, reward):
    #     self.Clients[self.getUniqueId(0, clientId)].registerReward(reward)

    def getScore(self, hostId, clientId):
        uniqueId = self.getUniqueId(hostId, clientId)
        return self.Clients[uniqueId].getScore()

    def getClientsInfo(self):
        clientInfo = {}
        for i, clientId in enumerate(self.Clients.keys()):
            client = self.Clients[clientId]
            clientInfo[client.clientId] = client.distance
        return clientInfo

    def nextClientIdToRun(self, hostId):
        init_id = hostId - 1
        lenPossible = len(self.feasibleClients)

        while True:
            clientId = str(self.feasibleClients[init_id])
            csize = self.Clients[clientId].size
            if csize >= self.filter_less and csize <= self.filter_more:
                return int(clientId)

            init_id = max(0, min(int(math.floor(self.rng.random() * lenPossible)), lenPossible - 1))

        return init_id

    def getUniqueId(self, hostId, clientId):
        return str(clientId)
        #return (str(hostId) + '_' + str(clientId))

    def clientSampler(self, clientId):
        return self.Clients[self.getUniqueId(0, clientId)].size

    def leaderOnHost(self, clientIds, hostId):
        self.leaderOnHosts[hostId] = clientIds
    def memberOnHost(self, clientIds, hostId):
        self.memberOnHosts[hostId] = clientIds
    # def correspLeaderofMemberOnHost(self, clientIds, hostId):
    #     self.correspLeaderofMemberOnHosts[hostId] = clientIds
    def clientLocalEpochOnHost(self, clientLocalEpochs, hostId):
        self.clientLocalEpochOnHosts[hostId] = clientLocalEpochs
    def memberLocalEpochOnHost(self, clientLocalEpochs, hostId):
        self.memberLocalEpochOnHosts[hostId] = clientLocalEpochs
    def clientDropoutratioOnHost(self, clientDropoutRatios, hostId):
        self.clientDropoutRatioOnHosts[hostId] = clientDropoutRatios

    def getCurrentLeaderIds(self, hostId):
        return self.leaderOnHosts[hostId]
    def getCurrentMemberIds(self, hostId):
        return self.memberOnHosts[hostId]
    # def getCurrentLeaderOfMemberIds(self, hostId):
    #     return self.correspLeaderofMemberOnHosts[hostId]
    def getCurrentClientLocalEpoch(self, hostId):
        return self.clientLocalEpochOnHosts[hostId]
    def getCurrentMemberLocalEpoch(self, hostId):
        return self.memberLocalEpochOnHosts[hostId]
    def getCurrentClientDropoutRatio(self, hostId):
        return self.clientDropoutRatioOnHosts[hostId]

    def getLeaderLenOnHost(self, hostId):
        return len(self.leaderOnHosts[hostId])
    def getMemberLenOnHost(self, hostId):
        return len(self.memberOnHosts[hostId])

    def getClientSize(self, clientId):
        return self.Clients[self.getUniqueId(0, clientId)].size

    def getSampleRatio(self, clientId, hostId, even=False):
        totalSampleInTraining = 0.

        if not even:
            for key in self.leaderOnHosts.keys():
                for client in self.leaderOnHosts[key]:
                    uniqueId = self.getUniqueId(key, client)
                    totalSampleInTraining += self.Clients[uniqueId].size

            #1./len(self.leaderOnHosts.keys())
            return float(self.Clients[self.getUniqueId(hostId, clientId)].size)/float(totalSampleInTraining)
        else:
            for key in self.leaderOnHosts.keys():
                totalSampleInTraining += len(self.leaderOnHosts[key])

            return 1./totalSampleInTraining
    # def getMemberSampleRatio():

    def getFeasibleClients(self, cur_time):
        if self.user_trace is None:
            return self.feasibleClients
        feasible_clients = []
        #logging.info("self.feasibleClients:{}".format(len(self.feasibleClients)))
        for clientId in self.feasibleClients: # len(self.feasibleClients) = 491
            if self.Clients[self.getUniqueId(0, clientId)].isActive(cur_time):
                feasible_clients.append(clientId)

        return feasible_clients

    def isClientActive(self, clientId, cur_time):
        return self.Clients[self.getUniqueId(0, clientId)].isActive(cur_time)

    def resampleClients(self, numOfClients, cur_time=0):
        self.count += 1

        feasible_clients = self.getFeasibleClients(cur_time) 
        logging.info("self.count:{},feasible_clients:{}".format(self.count, len(feasible_clients)))
        if len(feasible_clients) <= numOfClients:
            # if self.mode == "heaps":
            #     Leader_Clients = feasible_clients
                # Member_Clients = []
                # logging.info("Leader_Clients:{}, Member_Clients:{}".format(len(Leader_Clients), len(Member_Clients)))
                # return Leader_Clients, Member_Clients
            # else:
            return feasible_clients

        # pickled_clients = None
        # Clients = []
        # Leader_Clients = []
        # Member_Clients = []
        feasible_clients_set = set(feasible_clients)

        if self.count > 1:
            LeaderClients = self.ucbSampler.select_participant(numOfClients, feasible_clients=feasible_clients_set)
            for item in LeaderClients:    
                assert (item in feasible_clients_set)

        # elif self.mode == "heaps" and self.count >= 1:
        #     Leader_Clients, Member_Clients = self.ucbSampler.select_participant(numOfClients, feasible_clients=feasible_clients_set)
        #     logging.info("Leader_Clients:{}, Member_Clients:{}".format(len(Leader_Clients), len(Member_Clients)))
        #     for item in Leader_Clients:    
        #         assert (item in feasible_clients_set)
        #     for item in Member_Clients:    
        #         assert (item in feasible_clients_set)
        else:
            self.rng.shuffle(feasible_clients)
            client_len = min(numOfClients, len(feasible_clients) -1)
            LeaderClients = feasible_clients[:client_len]
            for item in LeaderClients:    
                assert (item in feasible_clients_set)

        # if self.mode == "heaps":
        #     return Leader_Clients, Member_Clients
        # else:
        return LeaderClients
    
    def resampleMembers(self, roundDuration, leadersToRun, cur_time):
        Member_Clients = []
        feasibleclients = self.getFeasibleClients(cur_time)
        # if len(feasibleclients) <= int(self.args.total_worker* self.args.overcommit*(1-self.args.leaderFactor)):
        #     Member_Clients = feasibleclients
        #     return Member_Clients
        feasible_clients_set = set(feasibleclients)

        # clientswithScore = self.ucbSampler.get_sortedClients()
        # availableClients = list(set(clientswithScore) | set(leadersToRun))
        # commuTime_Clients = {}
        # compuTime_Clients = {}
        # for client in availableClients:
        #     _,client_compu,client_commu = self.getCompletionTime(client, batch_size=self.args.batch_size, 
        #                         upload_epoch=self.args.upload_epoch, model_size=self.args.model_size * self.args.clock_factor)
        #     commuTime_Clients[client] = client_commu
        #     compuTime_Clients[client] = client_compu
        if self.count > 1:
            Member_Clients = self.ucbSampler.resample_Members(roundDuration, leadersToRun, cur_time, 
                        feasible_clients=feasible_clients_set) #, commuTimeofClients=self.commuTime_Clients, compuTimeofClients=self.compuTime_Clients
        for item in Member_Clients:    
            assert (item in feasible_clients_set)
        return Member_Clients
    
    def generateMigrationMatrix(self, leaders, members):
        return self.ucbSampler.getMigrationMatrix(leaders, members)
    
    def getAllMetrics(self):
        if self.mode == "oort":
            return self.ucbSampler.getAllMetrics()
        elif self.mode == "heaps":
            return self.ucbSampler.getAllMetrics()
        return {}

    def getDataInfo(self):
        return {'total_feasible_clients': len(self.feasibleClients), 'total_length': self.feasible_samples}

    def getClientGradient(self, clientId):
        if self.mode == "oort":
            return self.ucbSampler.get_client_metric(clientId)
        elif self.mode == "heaps":
            return self.ucbSampler.get_client_metric(clientId)
        else:
            feedbacks = {
                'reward': 1,
                'gradient': 0,
            }
            return feedbacks

    def get_median_reward(self):
        if self.mode == "oort":
            return self.ucbSampler.get_median_reward()
        elif self.mode == "heaps":
            return self.ucbSampler.get_median_reward()
        return 0.
