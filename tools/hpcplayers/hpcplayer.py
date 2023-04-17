# -*- coding=utf-8 -*-
# ------------------------------------------------------------------------
# Siamese DETR
# Copyright (c) 2023 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------


import os
import sys
import datetime
from tqdm import tqdm
import mpi4py.MPI as MPI


class HPCPlayer:
    def __init__(self, root, source, logfile):

        self.root = root
        self.source = source
        self.logfile = logfile

        self.world = MPI.COMM_WORLD
        self.rank = self.world.Get_rank()
        self.worldSize = self.world.Get_size()
        self.masterRank = 0

        if self.rank == self.masterRank:
            self.role = 'Master'
        else:
            self.role = 'Worker'

        if self.role == 'Master':
            self.print()
            self.print('##################################################################')
            self.print('############################ HPCPlayer ###########################')
            self.print('##################################################################')
            self.print()
            self.begin = datetime.datetime.now()
            self.masterSourceLineNums = self.sourceLinesCounting()
            self.masterSourceFileObj = open(self.source, 'r')
            if os.path.exists(self.logfile):
                raise AssertionError('logfile must be not exist in order to avoid miss-overwriting')
            self.masterLogFileObj = open(self.logfile, 'w')
            self.masterWorkLoadMonitor = [0] * self.worldSize
            self.pbar = tqdm(total=self.masterSourceLineNums)
            self.pbarRefreshFreq = 10

    @staticmethod
    def print(*content):
        print(*content)
        sys.stdout.flush()

    def sourceLinesCounting(self):
        self.print('Master counting source lines...')
        count = 0
        file = open(self.source, 'r')
        line = file.readline()
        while line:
            count += 1
            line = file.readline()
        file.close()
        self.print('Done, # of lines:', count)
        return count

    def MasterProcess(self):

        initializedWorker = set()
        workerWorking = [False] * self.worldSize
        readLinesCount = doneLinesCount = 0

        while readLinesCount < self.masterSourceLineNums or any(workerWorking):  
            for ithWorker in range(1, self.worldSize):
                if not workerWorking[ithWorker] and readLinesCount < self.masterSourceLineNums:  
                    line = self.masterSourceFileObj.readline().rstrip('\n')  
                    self.world.send(line, dest=ithWorker)
                    workerWorking[ithWorker] = True
                    readLinesCount += 1

            data = self.world.recv(source=MPI.ANY_SOURCE)
            receiveWorkerRank, receiveDoneLineResult = data
            self.masterLogFileObj.write(receiveDoneLineResult + '\n')  
            self.masterLogFileObj.flush()
            workerWorking[receiveWorkerRank] = False  
            initializedWorker.add(receiveWorkerRank)  
            self.masterWorkLoadMonitor[receiveWorkerRank] += 1
            doneLinesCount += 1
            info = f'[INFO] worker init: {len(initializedWorker)}/{self.worldSize-1}, Progress'
            self.pbar.set_description(info)
            if doneLinesCount % self.pbarRefreshFreq == 0:
                self.pbar.update(self.pbarRefreshFreq)

        self.pbar.close()
        self.masterLogFileObj.close()
        self.masterStopAllWorkers()  
        self.print(f'Elapsed Time: {datetime.datetime.now()-self.begin}')
        print('Workload table:', self.masterWorkLoadMonitor)

    def WorkerProcess(self):
        while True:
            receivedLine = self.world.recv(source=self.masterRank)
            if receivedLine is None:
                break
            receiveDoneLineResult = self.core(receivedLine)
            data2Send = (self.rank, receiveDoneLineResult)
            self.world.send(data2Send, dest=self.masterRank)
        self.workerEndingJobs()

    def run(self):
        if self.role == 'Master':
            self.MasterProcess()
        else:
            self.WorkerProcess()

    def masterStopAllWorkers(self):
        for ithWorker in range(1, self.worldSize):
            self.world.send(None, dest=ithWorker)

    def workerEndingJobs(self):
        pass

    def core(self, sourceLine):
        raise NotImplementedError
