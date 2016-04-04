from __future__ import division
from __future__ import print_function

import argparse
import shutil
import collections
import itertools
import math
import os
import pickle
import random
import time

import networkx as nx
import numpy as np
from BitVector import *
from lattice import Lattice
import matplotlib.pyplot as plt


class SentinelNoDb:

    def remove_folder(self,path):
    # check if folder exists
        if os.path.exists(path):
        # remove if exists
            shutil.rmtree(path)


    def createtable(self):

        self.warning("Start create table")
        dicttarget = {}

        while len(self.tuples) < self.nbcell:
            vector = []
            for d in range(self.nbdim):
                self.tvaldim.setdefault(d, set())
                rval = random.randint(1, self.nbvalperdim)
                vector.append(rval)
                self.tvaldim[d].add(str(rval))
            self.tuples.add(",".join(map(str, vector)))

        # print(self.tvaldim)
        # sys.exit()

        self.tuples = list(self.tuples)
        self.warning("Tuples are created")
        self.target = random.choice(self.tuples)
        vt = self.target.split(",")
        vtgood = []

        for el in vt:
            proba = random.randint(1, 100)
            if proba > 0:
                vtgood.append(str('*'))
            else:
                vtgood.append(el)
        self.target = vtgood
        self.warning("Target has been chosen ({0})".format(self.target))

        data = []
        rowid = 0

        for cell in self.tuples:
            print(rowid)
            tempseq = map(int, cell.split(","))
            for m in range(self.nbmeasure):
                ts = []
                choice = random.randint(1, 3)
                if choice == 1:
                    for t in range(self.nbtime): ts.append(int(random.randint(1, 100)))
                elif choice == 2:
                    for t in range(self.nbtime):
                        val = int(random.paretovariate(2))
                        if val <= 0:
                            ts.append(1)
                        else:
                            ts.append(val)
                else:
                    for t in range(self.nbtime):
                        val = int(random.gauss(30, 10))
                        if val <= 0:
                            ts.append(1)
                        else:
                            ts.append(val)
                tempseq.extend(ts)
                if m == self.nbmeasure - 1:
                    for i in range(self.nbtime) : self.aggrtarget[i] += ts[i]
            data.append(tempseq)

            for idDim in range(self.nbdim):
                self.index.setdefault(idDim,{})
                self.index[idDim].setdefault(tempseq[idDim],[])
                self.index[idDim][tempseq[idDim]].append(rowid)

            rowid += 1

        self.data = np.array(data)
        print(data)
        print(self.index)
        print(self.aggrtarget)

        dirstr = "{0}_{1}_{2}_{3}_{4}".format(self.nbdim, self.nbvalperdim, self.nbcell, self.nbtime, self.nbmeasure)
        pathdir = "./data/{0}".format(dirstr)
        self.remove_folder(pathdir)
        os.mkdir(pathdir)
        np.save(pathdir + "/data", self.data)
        with open(pathdir + "/vars", "wb") as f:
            pickle.dump(self.tvaldim, f)
            pickle.dump(self.tuples, f)
            pickle.dump(self.target, f)
            pickle.dump(self.index, f)
            pickle.dump(self.indexCells, f)
            pickle.dump(self.indextarget, f)
            pickle.dump(self.aggrtarget, f)



        """
            for t in range(1, self.nbtime + 1):
                vector = [t] + map(int, cell.split(","))
                dicttarget.setdefault(t, 0)
                for m in range(self.nbmeasure):
                    vector.append(tempseq[m][t - 1])
                    if (m == self.nbmeasure - 1): dicttarget[t] += tempseq[m][t - 1]

                if t <= (self.nbtime - self.wmax):
                    self.indexCells.setdefault(cell, [])
                    self.indexCells[cell].append(rowid)
                    for idDim in range(self.nbdim):
                        self.index.setdefault(idDim, {})
                        self.index[idDim].setdefault(vector[idDim + 1], [])
                        self.index[idDim][vector[idDim + 1]].append(rowid)
                self.indextarget.append(rowid)
                data.append(tuple(vector))
            rowid += 1
        od = collections.OrderedDict(sorted(dicttarget.items()))
        for k, v in od.iteritems():
            self.aggrtarget.append((k, v))

        print(self.data)
        self.data = np.array(data)

        dirstr = "{0}_{1}_{2}_{3}_{4}".format(self.nbdim, self.nbvalperdim, self.nbcell, self.nbtime, self.nbmeasure)
        pathdir = "./data/{0}".format(dirstr)

        os.mkdir(pathdir)
        np.save(pathdir + "/data", self.data)
        with open(pathdir + "/vars", "wb") as f:
            pickle.dump(self.tvaldim, f)
            pickle.dump(self.tuples, f)
            pickle.dump(self.target, f)
            pickle.dump(self.index, f)
            pickle.dump(self.indexCells, f)
            pickle.dump(self.indextarget, f)
            pickle.dump(self.aggrtarget, f)

    """

    def warning(self, *objs):
        print("DEBUG: ", *objs, file=sys.stderr)

    def calculatetargetvectors(self):

        data = self.data[self.indextarget]
        # print(data)
        data = data[data[:, 0].argsort()]
        # print(data)

        for w in range(1, self.wmax + 1):
            rows = []
            for key, group in itertools.groupby(data, lambda x: x[0]):

                if int(key) in range(1 + w, self.nbtime - self.wmax + w + 1):
                    sum = 0
                    for thing in group:
                        sum += int(thing[-1])
                    rows.append((key, sum))

            indmeasures2 = self.rowstovector(rows, w + 1, self.nbtime - self.wmax)
            self.targetbitvectors[w] = self.converttobitvectors(indmeasures2)



    def calculatetargetvectors2(self):

        for w in range(1, self.wmax + 1):
            indmeasures2 = self.rowstovector2(self.aggrtarget[w:w+self.windowsize])
            print(indmeasures2)
            self.targetbitvectors[w] = self.converttobitvectors(indmeasures2)



    def rowstovector2(self,tab):
        indications = []
        val = tab[0]
        for i in range(1,len(tab)):
            curr = tab[i]
            if (curr - val) / val >= 0.005:
                indications.append('1')
            elif (curr - val) / val <= -0.005:
                indications.append('-1')
            else:
                indications.append('0')
            val = curr

        return indications

    def rowstovector(self, records, deb, nbtime):
        currentvalues = 1
        indications = []
        currentindex = 0
        firstDate = int(records[0][0])
        if firstDate == deb:
            currentvalues = int(records[0][1])
            currentindex = 1

        currentvalues = int(records[0][1])

        for t in range(1, nbtime):
            time = t + deb
            if currentindex >= len(records):  # no more value, fill the blanks with 0
                indications.append('0')
            else:  # there are still value(s)
                if time < int(records[currentindex][0]):  # not the good one
                    indications.append('0')
                else:
                    latest = int(currentvalues)
                    current = int(records[currentindex][1])
                    if (current - latest) / latest >= 0.005:
                        indications.append('1')
                    elif (current - latest) / latest <= -0.005:
                        indications.append('-1')
                    else:
                        indications.append('0')

                    currentvalues = current
                    currentindex += 1

        # print(indications)
        return indications
        # strindications = [",".join(indic) for indic in indications]
        # return strindications

    def converttobitvectors(self, vind):

        res = []
        tp = []
        tv = []
        tn = []
        for v in vind:
            if v == "0":
                tp.append(0)
                tn.append(0)
                tv.append(1)
            elif v == "1":
                tp.append(1)
                tv.append(0)
                tn.append(0)
            else:
                tp.append(0)
                tv.append(0)
                tn.append(1)

        res.append(BitVector(bitlist=tp))
        res.append(BitVector(bitlist=tv))
        res.append(BitVector(bitlist=tn))
        return res

    def calculateaggregate(self, mes, db, w1, w2, w):
        # print("dans calc")
        # print(db)

        def bitvectorstoind(bv1, bv2, bv3):
            r = []
            for i in range(bv1.length()):
                if bv1[i] == 1:
                    r.append(1)
                elif bv2[i] == 1:
                    r.append(0)
                elif bv3[i] == 1:
                    r.append(-1)
                else:
                    r.append('x')
            return r

        t1 = time.time()
        # self.warning("db => {0}".format(db))
        data = self.data[list(db)]
        data = data[data[:, 0].argsort()]
        # print(data)
        rows = []
        # sys.exit()
        for key, group in itertools.groupby(data, lambda x: x[0]):
            sum = 0

            for thing in group:
                # print "A %s is a %s." % (thing, key)
                sum += int(thing[self.nbdim + mes])
            # print sum
            rows.append((key, sum))
        if len(rows) == 0: return {}
        indmeasures = self.rowstovector(rows, w1, w2)
        # print("Inde measure")
        # print(indmeasures)
        bv = self.converttobitvectors(indmeasures)

        v1 = bv[0] & self.targetbitvectors[w][0]
        v2 = bv[1] & self.targetbitvectors[w][1]
        v3 = bv[2] & self.targetbitvectors[w][2]

        v4 = bv[2] & self.targetbitvectors[w][0]
        v5 = bv[1] & self.targetbitvectors[w][1]
        v6 = bv[0] & self.targetbitvectors[w][2]

        # print(v1)
        # print(v2)
        # print(self.targetbitvectors[w][0])
        # print(self.targetbitvectors[w][1])
        # print("------------------------------")

        cons1 = (v1.count_bits() + v2.count_bits() + v3.count_bits()) / v1.length();  # print("cons1={0}".format(cons1))
        cons2 = (v4.count_bits() + v5.count_bits() + v6.count_bits()) / v4.length();  # print("cons2={0}".format(cons2))

        self.calcaggr += (time.time() - t1)
        del data

        if cons1 >= self.mincons:
            # self.warning("+ => {0}".format(",".join(map(str,bitvectorstoind(v1, v2, v3)))))
            return (",".join(map(str, bitvectorstoind(v1, v2, v3))), '+', True)
        elif cons2 >= self.mincons:
            # self.warning("- => {0}".format(",".join(map(str,bitvectorstoind(v4, v5, v6)))))
            return (",".join(map(str, bitvectorstoind(v4, v5, v6))), '-', True)
        else:
            return (None, None, False)

    def calculateaggregate2(self, mes, db,w):
        # print("dans calc")
        #print(db)

        def bitvectorstoind(bv1, bv2, bv3):
            r = []
            for i in range(bv1.length()):
                if bv1[i] == 1:
                    r.append(1)
                elif bv2[i] == 1:
                    r.append(0)
                elif bv3[i] == 1:
                    r.append(-1)
                else:
                    r.append('x')
            return r

        t1 = time.time()
        # self.warning("db => {0}".format(db))
        data = self.data[list(db)]

        #print(data[0])
        #print(data.shape[0])

        begin = self.nbdim+(mes*self.nbtime)
        end = begin+self.windowsize
        rows = data[0][begin:end]
        for i in range(1,data.shape[0]):
            rows = np.add(rows,data[i][begin:end])

        if len(rows) == 0: return {}
        indmeasures = self.rowstovector2(rows)
        #print("Inde measure")
        #print(indmeasures)
        bv = self.converttobitvectors(indmeasures)

        v1 = bv[0] & self.targetbitvectors[w][0]
        v2 = bv[1] & self.targetbitvectors[w][1]
        v3 = bv[2] & self.targetbitvectors[w][2]

        v4 = bv[2] & self.targetbitvectors[w][0]
        v5 = bv[1] & self.targetbitvectors[w][1]
        v6 = bv[0] & self.targetbitvectors[w][2]

        # print(v1)
        # print(v2)
        # print(self.targetbitvectors[w][0])
        # print(self.targetbitvectors[w][1])
        # print("------------------------------")

        cons1 = (v1.count_bits() + v2.count_bits() + v3.count_bits()) / v1.length();  # print("cons1={0}".format(cons1))
        cons2 = (v4.count_bits() + v5.count_bits() + v6.count_bits()) / v4.length();  # print("cons2={0}".format(cons2))

        self.calcaggr += (time.time() - t1)
        del data

        if cons1 >= self.mincons:
            # self.warning("+ => {0}".format(",".join(map(str,bitvectorstoind(v1, v2, v3)))))
            return (",".join(map(str, bitvectorstoind(v1, v2, v3))), '+', True)
        elif cons2 >= self.mincons:
            # self.warning("- => {0}".format(",".join(map(str,bitvectorstoind(v4, v5, v6)))))
            return (",".join(map(str, bitvectorstoind(v4, v5, v6))), '-', True)
        else:
            return (None, None, False)





    def getdb(self, c, indices, offset):

        # print(self.index)
        t1 = time.time()
        # listIntersect = set(indices)
        # self.tconvert += time.time() - t1
        listIntersect = indices

        all = True

        for d in range(offset, len(c)):
            if c[d] != '*':
                taccess = time.time()
                tind = self.index[d][int(c[d])]
                self.taccess += time.time() - taccess
                # print("before : {0}".format(listIntersect))
                # self.warning ("d:{0} / c[d]:{1}".format(d,c[d]))
                if all:
                    all = False
                    listIntersect = set(tind)
                else:
                    # self.warning ("tind:{0}".format(tind))
                    tinter = time.time()
                    if len(tind) > 0: listIntersect = set(tind).intersection(listIntersect)
                    # if len(tind) > 0: listIntersect = np.intersect1d(listIntersect,tind)
                    # if len(tind) > 0: listIntersect = filter(set(tind).__contains__, listIntersect)

                    self.tinter += time.time() - tinter

        self.tgetdb += (time.time() - t1)
        if all:
            return indices
        else:
            return listIntersect

    def calculateupperbound(self, c, db):
        t1 = time.time()
        data = self.data[list(db)]
        ub = []
        listdistinctvalues = []
        # print(np.shape(db))
        for d in range(len(c)):
            listdistinctvalues.append(np.unique(data[:, d ]))
            # listdistinctvalues.append(np.unique(data[:, d]))
            if listdistinctvalues[-1].size > 1:
                ub.append(c[d])
            else:
                ub.append(listdistinctvalues[-1][0])
        self.upperbound += (time.time() - t1)
        del data
        return (ub, listdistinctvalues)

    def insertintotempclasses(self, c, ub, pid, bid, aggr, mes, w, sign):
        # self.warning("insert [{0} {1} {2} {3} {4} {5} {6} {7}]".format(bid, ub, c, pid, aggr, mes, w,sign))
        self.temporaryclasses.append(
            {'id': bid, 'ub': ub, 'lb': c, 'chd': pid, 'aggr': aggr, 'mes': mes, 'w': w, 'sign': sign})

    def dfs(self, c, db, k, pid, mes, w):
        print(c)
        (vectoraggr, sign, ok) = self.calculateaggregate2(mes, db, w)

        # 2) Compute the upper bound d of c by jumping to the appropriate upper bounds
        self.id += 1
        (ub, distinctvalues) = self.calculateupperbound(c, db)
        # self.warning("ub => {0}".format(ub))
        if c == self.tall: self.warning("distinct values => {0}".format(distinctvalues))
        if ok:
            # self.warning("insert {0}".format(vectoraggr))
            #self.insertintotempclasses(c, ub, pid, self.id, vectoraggr, mes, w, sign)
            self.insertingraph(vectoraggr, ub, w, sign, mes)

        # 3) If there is some j < k st c[j] = '*' and ub[j] != '*' => return
        for j in range(k):
            # print(j)
            if c[j] == '*' and ub[j] != '*': return

        # 4) for each k < j <= n s.t. d[j] = '*'
        for j in range(k + 1, self.nbdim):
            if ub[j] == '*':
                # distinctvalues = self.getdistinctvalues(c,j,1,self.nbtime -self.wmax)
                for val in distinctvalues[j]:
                    cell = list(c)

                    cell[j] = val
                    # print(cell)
                    db2 = self.getdb(cell, db, j)
                    if len(db2) > 0: self.dfs(cell, db2, j, self.id, mes, w)

    def up(self, v):

        convert = v.split(",")
        # print "down1alone"
        # print convert
        res = set()
        for idel in range(len(convert)):
            if convert[idel] != "*":
                toadd = convert[:idel] + ['*'] + convert[idel + 1:]
                toadd = map(str, toadd)
                res.add(",".join(toadd))
        return res

    def down(self, v):
        convert = v.split(",")
        # print("down")
        # print(v)
        # print convert
        res = set()
        for idel in range(len(convert)):
            if convert[idel] == "*":
                # self.warning("CHild de la dim {0} => {1}".format(idel, self.tvaldim[idel]))
                for v in self.tvaldim[idel]:
                    toadd = convert[:idel] + [v] + convert[idel + 1:]
                    toadd = map(str, toadd)
                    res.add(",".join(toadd))
        # print(res)
        return res

    def findmostspecific(self, w, mes):

        count = 1
        self.warning("[FIND MOST SPECIFIC (w={0})]".format(w))
        # self.warning("   M{0}".format(idmeasure))
        candidates = set()
        candidates = set(self.tuples)
        useless = set()
        level = 1

        while level <= (self.nbdim + 1):
            # twhile = time.time()
            # self.warning("\t[LEVEL {0} size candidates= {1}]".format(level,len(candidates)))

            newcand = set()
            for cand in candidates:
                print(str(count) + "_" + cand)

                if cand in useless:
                    useless.remove(cand)
                    useless.update(self.up(cand))
                else:

                    db = self.getdb(cand.split(','), self.sallindices, 0)
                    #(vectoraggr, sign, ok) = self.calculateaggregate2(mes, db, 1, self.nbtime - self.wmax, w)
                    (vectoraggr, sign, ok) = self.calculateaggregate2(mes, db,w)
                    if ok:
                        # self.warning("Ajout de {0} avec {1}".format(cand, vectoraggr))
                        self.insertingraph(vectoraggr, cand, w, sign, mes)
                        useless.update(self.up(cand))
                    else:
                        newcand.update(self.up(cand))
                count += 1

            level += 1
            candidates = newcand.copy()

    def findmostgeneral(self, w, mes):

        self.warning("[FIND MOST GENERAL (w={0})]".format(w))

        candidates = set()
        candidates.add(','.join(self.tall))
        useless = set()
        level = self.nbdim + 1
        while level > 0:

            # self.warning("\t[LEVEL {0} size candidates= {1}]".format(level,len(candidates)))

            newcand = set()
            for cand in candidates:
                print(cand)

                if cand in useless:
                    useless.remove(cand)
                    useless.update(self.down(cand))
                else:

                    db = self.getdb(cand.split(','), self.sallindices, 0)
                    if len(db) > 0:
                        (vectoraggr, sign, ok) = self.calculateaggregate2(mes, db, w)
                        if ok:
                            # self.warning("Ajout de {0} avec {1}".format(cand, vectoraggr))
                            self.insertingraph(vectoraggr, cand, w, sign, mes)
                            useless.update(self.down(cand))
                        else:
                            newcand.update(self.down(cand))
            # print(newcand)

            level -= 1
            candidates = newcand.copy()

    def getlv(self,agg):
            t=agg.split(",")
            lv=0
            for e in t:
                if e == 'x':lv +=1
            return lv


    def insertingraph(self, agg, ub, w, sign, mes):

        def fromaggtobit(agg):
            t = agg.split(",")
            r = []
            for e in t :
                if e =='x':r.append('0')
                else: r.append('1')
            return ",".join(r)



        t1 = time.time()
        lv = self.getlv(agg)
        strtoinsert = sign + "_" + ",".join(map(str, ub)) + "_" + str(mes) + "_" + str(w)

        bv = fromaggtobit(agg)

        if agg in self.indexvertices:
            self.indexvertices[bv].add(strtoinsert)
        else:

            # print("creation de {0}".format(agg))
            self.indexvertices[bv] = set()
            self.indexvertices[bv].add(strtoinsert)
            self.g.add_node(bv)
            lv = self.getlv(agg)
            self.nodesperlevel.setdefault(lv,set())

            self.nodesperlevel[lv].add(bv)
            self.mappingagg[bv]=agg
            self.sources.add(bv)
            #borders = getborders(agg)
            #print(borders)

            #self.createedges(agg,lv)
        self.timeinsert += time.time() - t1



    def fromaggtobitvector(self,agg):
        t = []
        ta = agg.split(',')
        for e in ta :
            if e == 'x':
                t.append(0)
            else :
                t.append(1)
        return BitVector(bitlist=t)




    def createedges(self, agg,lv):


        def issrc(agg):
            res = False
            notsrc = set()
            for e in self.sources:
                if self.islink(agg,e):
                    self.g.add_edge(agg,e)
                    notsrc.add(e)
                    issrc=True

            if res :
                self.sources = self.sources - notsrc
                self.sources.add(agg)
            return res

        ordlv = sorted(self.nodesperlevel,reverse=True)

        t1 = time.time()
        notsrc = set()
        useless = set()
        for idlv in ordlv :
            nodes = self.nodesperlevel[idlv]
            if idlv < lv :
                for n in nodes :
                    if self.islink(agg,n):
                        self.g.add_edge(agg,n)
                        notsrc.add(n)
            elif idlv > lv :
                for n in nodes :
                    if n not in useless:
                        if self.islink(n,agg):
                            pred = nx.ancestors(self.g,n)
                            useless = useless | pred
                            self.g.add_edge(n,agg)
                            notsrc.add(agg)

        self.sources = self.sources - notsrc
        self.nodesperlevel[lv].add(agg)
        issrc= False
        if agg not in notsrc:
            issrc = True
        self.tcreationedge += (time.time() - t1)
        t2=time.time()



        if issrc:
            notsrc.clear()
            toadd = set()
            for s in self.sources:
                if s != agg :
                    res = self.getancestor(agg,s)
                    if len(res) > 0 :
                        self.g.add_edge(res[0],agg)
                        self.g.add_edge(res[0],s)
                        notsrc.add(agg)
                        notsrc.add(s)
                        toadd.add(res[0])
                        lvsrc = self.getlv(res[0])
                        self.nodesperlevel.setdefault(lvsrc,set())
                        self.nodesperlevel[lvsrc].add(res[0])
            self.sources = self.sources - notsrc
            self.sources = self.sources | toadd

        self.tupdatesrc += (time.time() - t2)



    def createedgesfinal(self):


        ordlv = sorted(self.nodesperlevel,reverse=False)

        print("step1")
        print(ordlv)

        for lv in ordlv:
            print("lv {0} contient {1} noeuds".format(lv,len(self.nodesperlevel[lv])))


        #sys.exit()
        # step 1

        notsrc = set()
        nbedges = 0

        for i in range(len(ordlv)-1):
            lva = ordlv[i]
            nodesa = self.nodesperlevel[lva]
            print("Lv {0} (size={1})".format(lva,len(nodesa)))
            for j in range(i+1,len(ordlv)):
                lvb = ordlv[j]
                nodesb = self.nodesperlevel[lvb]
                print("\t - Lv {0} (size={1})".format(lvb,len(nodesb)))

                if lva+lvb > self.nbauthorizedx:
                    for n1 in nodesa:
                        for n2 in nodesb:
                            if self.islink(n1,n2):
                                self.g.add_edge(n1,n2)
                                notsrc.add(n2)
                                nbedges += 1
                else :
                    for n1 in nodesa:
                        for n2 in nodesb:
                            self.g.add_edge(n1,n2)
                            notsrc.add(n2)
                            nbedges += 1


        self.sources = self.sources - notsrc

        print("{0} edges created".format(nbedges))
        sys.exit()

        print("step2")
        print(len(self.sources))

        # step 2
        notsrc.clear()
        toadd = set()
        lsources = list(self.sources)
        for i in range(len(lsources)-1):
            srca = lsources[i]
            for j in range(i+1,len(lsources)):
                srcb = lsources[j]
                r = self.getancestor(srca,srcb)
                if len(r) > 0:
                    self.g.add_edge(r[0],srca)
                    self.g.add_edge(r[0],srcb)
                    notsrc.add(srca)
                    notsrc.add(srcb)
                    toadd.add(r[0])


        self.sources = self.sources - notsrc
        self.sources = self.sources | toadd

        print("step3")
        print(len(toadd))

        lsources = list(toadd)
        todel = set()
        for i in range(len(lsources)-1):
            a = lsources[i]
            for j in range(i+1,len(lsources)):
                b=lsources[j]
                if self.islink(a,b):
                    todel.add(a)
                    self.g.add_edge(a,b)
                elif self.islink(b,a):
                    todel.add(b)
                    self.g.add_edge(b,a)
        self.sources = self.sources - todel




    def createedgesfinal2(self):


        def getcommon(a,b) :
            va = BitVector(bitlist=map(int,a.split(",")))
            vb = BitVector(bitlist=map(int,b.split(",")))

            return va&vb


        ordlv = sorted(self.nodesperlevel,reverse=False)

        print("step1")
        print(ordlv)

        for lv in ordlv:
            print("lv {0} contient {1} noeuds".format(lv,len(self.nodesperlevel[lv])))


        #sys.exit()
        # step 1

        notsrc = set()
        nbedges = 0





        for i in range(len(ordlv)-1):
            lva = ordlv[i]
            nodesa = self.nodesperlevel[lva]
            print("Lv {0} (size={1})".format(lva,len(nodesa)))
            for j in range(i,len(ordlv)):
                lvb = ordlv[j]
                nodesb = self.nodesperlevel[lvb]
                print("\t - Lv {0} (size={1})".format(lvb,len(nodesb)))
                for n1 in nodesa:
                    for n2 in nodesb:
                        common = getcommon(n1,n2)

                        if str(common)==n2:
                            #self.g.add_edge(n2,n1)
                            notsrc.add(n1)

                        elif common.count_bits() >= self.nbminnonx:
                        # il faut creer le noeud (si besoin) et l'inserer dans le niveau et les arcs
                            self.indexvertices.setdefault(str(common),set())
                            #self.g.add_edge(str(common),n1)
                            #self.g.add_edge(str(common),n2)
                            lv = common.length() - common.count_bits()
                            self.nodesperlevel.setdefault(lv,set())

                ordlv = sorted(self.nodesperlevel,reverse=False)

        print("nb nodes after=> {0}".format(len(self.g.nodes())))


    def islink(self,a, b):
        # print(a)
        # print(b)
        ta = a.split(",")
        tb = b.split(",")
        # print(ta)
        # print(tb)
        count = 0
        while count < len(ta) and (ta[count] == 'x' or (ta[count] == tb[count])): count += 1
        return len(ta) == count


    def createedges2(self):
        todeltosources = set()
        listnodes = self.g.nodes()
        for i in range(len(listnodes)):
            nodeA = listnodes[i]
            for j in range(i+1,len(listnodes)):
                nodeB = listnodes[j]
                if self.islink(nodeA,nodeB):
                    todeltosources.add(nodeB)
                    self.g.add_edge(nodeA,nodeB)
                elif self.islink(nodeB,nodeA):
                    todeltosources.add(nodeA)
                    self.g.add_edge(nodeB,nodeA)

        self.sources = self.sources - todeltosources
        print(len(self.sources))

        #"""
        pos=nx.spring_layout(self.g)

        nx.draw_networkx_nodes(self.g,pos,
                               node_color='r',
                               node_size=500,
                           alpha=0.8)

        nx.draw_networkx_edges(self.g,pos,width=1.0,alpha=0.5)
        labels=nx.draw_networkx_labels(self.g,pos=pos,font_size=16)
        plt.axis('off')
        plt.savefig("ex.png")
        plt.show()
        #"""

    def getancestor(self,a,b):

        def findx(t):
            r = set()
            for i in range(len(t)):
                if t[i] == "x" : r.add(i)
            return r

        def findnonx(t):
            r = set()
            for i in range(len(t)):
                if t[i] != "x" : r.add(i)
            return r

        def fromindicestoborder(t,s):
            r  = []
            for i in range(len(t)):
                if i in s: r.append('x')
                else : r.append(t[i])
            return r


        ta = a.split(",")
        tb = b.split(",")

        sax = findx(ta)
        sbx = findx(tb)

        sx = sax | sbx

        #print("nbauthorized {0} / nbreal {1}".format(self.nbauthorizedx,len(sx)))

        r = []
        if len(sx) > self.nbauthorizedx : return r
        else : r.append(",".join(fromindicestoborder(ta,sx))); return r

        """
        elif len(sx) == self.nbauthorizedx : r.append(",".join(fromindicestoborder(ta,sx))); return r
        else :
            tborder = fromindicestoborder(ta,sx)
            snxborder = findnonx(tborder)
            nbxtoadd = self.nbauthorizedx - len(sx)
            print("{0} x to add in {1} elements".format(nbxtoadd,len(snxborder)))
            #for el in itertools.combinations(snxborder,nbxtoadd):
            #    r.append(",".join(fromindicestoborder(ta,list(el))))
            r.append(",".join(fromindicestoborder(ta,sx)))
            return r
        """


    def filtersources2(self):

        cand = self.sources
        sourcestodrop = set()
        round = 1
        while len(cand)>1:
            print("Round {0} => size = {1}".format(round,len(cand)))
            newcand = set()
            src = list(cand)
            for i in range(len(src) - 1):
                a = src[i]
                for j in range(i+1,len(src)):
                    b = src[j]
                    borders = self.getancestor(a,b)
                    if len(borders)>0 :
                        sourcestodrop.add(a)
                        sourcestodrop.add(b)
                        newcand.add(borders[0])
            cand = newcand



    def filtersources(self):

        lsources = list(self.sources)
        todel = set()
        for i in range(len(lsources)-1):
            a = lsources[i]
            for j in range(i+1,len(lsources)):
                b=lsources[j]
                if self.islink(a,b):
                    todel.add(a)
                    self.g.add_edge(a,b)
                elif self.islink(b,a):
                    todel.add(b)
                    self.g.add_edge(b,a)
        self.sources = self.sources - todel

    def union(self, a,b):
        ta = a.split(",")
        tb = b.split(",")
        tr = []
        for i in range(len(ta)):
            if ta[i] != 'x': tr.append(ta[i])
            elif tb[i] != 'x': tr.append(tb[i])
            else: tr.append('x')
        return ",".join(tr)


    def intersection(self, a,b):
        ta = a.split(",")
        tb = b.split(",")
        tr = []
        for i in range(len(ta)):
            if ta[i] != 'x' and tb[i] != 'x': tr.append(ta[i])
            else: tr.append('x')
        return ",".join(tr)



    def __init__(self, nbdim, nbvalperdim, nbmeasure, nbtime, nbcell, wmax, minharm, mincons, type):
        # if __name__ == "__main__":
        self.nbdim = nbdim[0]
        self.nbvalperdim = nbvalperdim[0]
        self.nbmeasure = nbmeasure[0]
        self.nbtime = nbtime[0]
        self.nbcell = nbcell[0]
        self.wmax = wmax[0]
        self.type = type[0]
        self.datatablename = "data"
        self.dbname = "qctree_d{0}_{1}_c{4}_m{2}_t{3}_h{5}_C{6}_w{7}".format(nbdim, nbvalperdim,
                                                                             nbmeasure, nbtime, nbcell, minharm,
                                                                             mincons, wmax)
        self.temporaryclasses = []
        self.id = -1
        # self.dictprimenumber = {}   # prime->value
        # self.revertdictprimenumber = {} # value->prime
        # self.multipledim = {}
        # self.primemappingdim = {}
        # self.valall = 1

        # self.generateprimenumbers()
        # self.primereprbasecells = set()

        # self.conn = sqlite3.connect(self.dbpath+self.dbname)
        self.dictmapp = {}

        self.minharm = minharm[0]
        self.mincons = mincons[0]

        self.target = []
        self.bittarget = []
        self.tuples = set()
        self.length = 0
        self.data = None
        self.calcaggr = 0
        self.upperbound = 0
        self.tgetdb = 0
        self.timeinsert = 0
        self.tinter = 0
        self.taccess = 0
        self.tconvert = 0
        self.tcreationedge = 0
        self.tupdatesrc = 0
        self.index = {}
        self.indexCells = {}
        self.indextarget = []
        self.tvaldim = {}
        self.targetchild = set()
        self.targetbitvectors = {}
        self.subtables = {}  # w -> measures -> data
        # self.allindices = np.arange((self.nbtime-self.wmax) * self.nbcell)
        self.allindices = np.arange(self.nbcell)
        self.sallindices = set(self.allindices)

        self.windowsize = self.nbtime - self.wmax


        self.aggrtarget = np.zeros(self.nbtime)

        self.g = nx.DiGraph()
        self.sources = set()
        self.indexvertices = {}
        self.nodesperlevel = {}
        self.mappingagg = {}

        self.stringdim = "D1"
        for d in range(2, self.nbdim + 1):
            self.stringdim += ",D{0}".format(d)


            # if not self.tableexist("mgr") :

            # if not self.tableexist("data"):
            # self.db  = Base('data')

            # pprint.pprint(self.index)

        dirstr = "{0}_{1}_{2}_{3}_{4}".format(self.nbdim, self.nbvalperdim, self.nbcell, self.nbtime, self.nbmeasure)
        pathdir = "./data/{0}".format(dirstr)

        self.createtable()

        if os.path.exists(pathdir):
            self.data = np.load(pathdir + "/data.npy")
            with open(pathdir + "/vars", "r") as f:
                self.tvaldim = pickle.load(f)
                self.tuples = pickle.load(f)
                self.target = pickle.load(f)
                self.index = pickle.load(f)
                self.indexCells = pickle.load(f)
                self.indextarget = pickle.load(f)
                self.aggrtarget = pickle.load(f)
                self.warning("Table reloaded....")
        else:
            self.createtable()
            self.warning("Table created....")

        print(self.data)
        self.calculatetargetvectors2()
        self.warning("Target vectors created....")
        self.nbauthorizedx = int(math.floor(self.targetbitvectors[1][0].length() * (1- self.minharm)))
        self.nbminnonx = int(math.floor(self.targetbitvectors[1][0].length() * self.minharm))

        print(self.nbauthorizedx)
        #sys.exit()


        # self.down("*,*,*,*,*,*,*,*,*,*")
        # sys.exit()


        self.tall = ['*'] * self.nbdim

        t1 = time.time()
        for m in range(1, self.nbmeasure):
            self.warning("\t Dealing with M{0}".format(m))
            for w in range(1, self.wmax + 1):
                self.warning("\t\t Dealing with w = {0}".format(w))

                if self.type == "qc":
                    self.dfs(self.tall, self.allindices, -1, self.id, m, w)
                elif self.type == "ms":
                    self.findmostspecific(w, m)
                elif self.type == "mg":
                    self.findmostgeneral(w, m)

        print("NODES")
        print(len(self.g.nodes()))

        self.L = Lattice(self.indexvertices.keys(), self.union,self.intersection)
        #print(self.L.Uelements)
        #print(self.L.BottomElement)
        #sys.exit()


        #for e in self.g.nodes():
        #    if self.g.in_degree(e) == 0: self.sources.add(e)

        #print("createedges")
        #self.createedges2()
        print("filtersources")

        #self.filtersources()
        self.createedgesfinal2()

        """

        pos=nx.shell_layout(self.g)

        nx.draw_networkx_nodes(self.g,pos,
                               node_color='r',
                               node_size=500,
                           alpha=0.8)

        nx.draw_networkx_edges(self.g,pos,width=1.0,alpha=0.5)
        labels=nx.draw_networkx_labels(self.g,pos=pos,font_size=16)
        plt.axis('off')
        plt.savefig("ex.png")
        plt.show()
        """


        self.warning('Time => {0} sec'.format(time.time() - t1))
        self.warning('Aggr => {0} sec'.format(self.calcaggr))
        self.warning('TimeInsert => {0} sec'.format(self.timeinsert))
        self.warning('\tcreationedge => {0} sec'.format(self.tcreationedge))
        self.warning('\tupdatesrc => {0} sec'.format(self.tupdatesrc))
        self.warning('getDb => {0} sec'.format(self.tgetdb))
        self.warning('\tconversion => {0} sec'.format(self.tconvert))
        self.warning('\tintersection => {0} sec'.format(self.tinter))
        self.warning('\taccess => {0} sec'.format(self.taccess))

    """
        for w in range(1, self.wmax + 1):
            self.gettargetbitvectors(w)
            self.length = self.bittarget[0].length()
            self.findmostgeneral(w)
            self.findmostspecific(w)
            #sys.exit()
            self.createmapping()
            self.insertitems(w)
            self.warning(self.dbname)
    """


if __name__ == "__main__":
    """
    nbdim = int(sys.argv[1])
    nbvalperdim = int(sys.argv[2])
    nbmeasure = int(sys.argv[3])
    nbtime = int(sys.argv[4])
    nbtuplepertime = int(sys.argv[5])
    nbcell = int(sys.argv[6])
    wmax = int(sys.argv[7])
    minharm = float(sys.argv[8])
    mincons = float(sys.argv[9])
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('-dim', nargs=1, type=int)
    parser.add_argument('-val', nargs=1, type=int)
    parser.add_argument('-mes', nargs=1, type=int)
    parser.add_argument('-time', nargs=1, type=int)
    parser.add_argument('-cells', nargs=1, type=int)
    parser.add_argument('-wmax', nargs=1, type=int)
    parser.add_argument('-harm', nargs=1, type=float)
    parser.add_argument('-cons', nargs=1, type=float)
    parser.add_argument('-type', nargs=1)

    args = parser.parse_args()

    sent = SentinelNoDb(args.dim, args.val, args.mes, args.time, args.cells, args.wmax, args.harm, args.cons, args.type)
