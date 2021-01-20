import pandas as pd
import math
import numpy as np
from DataSet import DataSet
from datetime import date
class SportDataSet:

    def __init__(self, season, game):

        self.playersToI={}
        self.ItoPlayers={}
        self.matches={}
        self.season=season
        self.odds={}
        self.game=game
        self.data=None

        if game=="F":
            self.buildF()
        elif game=="H":
            self.buildHNoDraw()

    def buildHNoDraw(self):
        self.dataFrame = pd.read_excel("NHL/nhl odds "+self.season+".xlsx")
        data = self.dataFrame

        dbf=data["Date"]
        dates=[]
        years=self.season.split("-")
        year=years[0]
        for dat in dbf:
            month=int(str(dat)[:-2])
            day=int(str(dat)[-2:])
            if month<7:
                year=years[1]
            dates.append(date(int(year), month, day))

        firstDay=dates[0];
        lastDay=dates[-1];
        totalDays=(lastDay-firstDay).days+1
        print(totalDays)


        for i in data["Team"]:
            if i not in self.playersToI.keys():
                self.playersToI[i] = len(self.playersToI)
                self.ItoPlayers[len(self.ItoPlayers)] = i


        xData=[]
        yData=[]
        oddsData=[]

        for i in range(totalDays):
            xData.append([])
            yData.append([])
            oddsData.append([])

        #xData = np.zeros([int(len(data.index)/2),1 ,len(self.playersToI)])
        #yData = np.zeros([int(len(data.index)/2),1 ,2])

        for i in range(0, len(data), 2):
            name = "match#" + str((i + 1) - int((i + 1) / 2))
            home = data["Team"][i + 1]
            away = data["Team"][i]
            gA = data["1st"][i] + data["2nd"][i] + data["3rd"][i]
            gH = data["1st"][i + 1] + data["2nd"][i + 1] + data["3rd"][i + 1]
            finalA = data["Final"][i]
            finalH = data["Final"][i + 1]

            oddsA = data["Open"][i]
            oddsH = data["Open"][i + 1]

            dat=data["Date"][i]
            month=int(str(dat)[:-2])
            day=int(str(dat)[-2:])
            if month<8:
                year=years[1]
            else:
                year=years[0]
            matchDate=(date(int(year), month, day))
            dayIndex=(matchDate-firstDay).days

            hI = self.playersToI[home];
            aI = self.playersToI[away]

            #index=(i) - int((i + 1) / 2)

            xij=np.zeros(len(self.playersToI))
            yij = np.zeros(2)
            xij[hI]=1
            xij[aI]=-1

            #xData[index][0][hI] = 1;
            #xData[index][0][aI] = -1

            if finalH>finalA:
                #yData[index][0][0] = 1
                yij[0]=1
            else:
                #yData[index][0][1] = 1
                yij[1]=1

            if finalH==finalA:
                print("draw warning draw warning draw warning draw warning")

            odds=[oddsH, oddsA]
            #self.odds[i] = odds
            self.matches[name] = len(self.matches)

            xData[dayIndex].append(xij)
            yData[dayIndex].append(yij)
            oddsData.append(odds)
            self.odds=oddsData

        self.data = DataSet(xData, yData, len(self.playersToI), 2)


    def buildF(self):
        self.dataFrame = pd.read_csv("EPL/"+self.season+".csv")
        data = self.dataFrame

        for i in data["HomeTeam"]:
            if i not in self.playersToI.keys():
                self.playersToI[i]=len(self.playersToI)
                self.ItoPlayers[len(self.ItoPlayers)]=i

        xData=np.zeros([len(data.index),1 ,len(self.playersToI)])
        yData=np.zeros([len(data.index),1 ,3])

        for i in data.index:
            name="match#"+str(i+1)
            home=data["HomeTeam"][i]
            gH=data["FTHG"][i]
            away=data["AwayTeam"][i]
            gA=data["FTAG"][i]
            status=data["FTR"][i]

            hI=self.playersToI[home]; aI=self.playersToI[away]
            xData[i][0][hI]=1; xData[i][0][aI]=-1

            if status == "H":
                yData[i][0][0] = 1
            elif status == "A":
                yData[i][0][1] = 1
            elif status == "D":
                yData[i][0][2] = 1

            self.odds[i]=[data["B365H"][i],data["B365A"][i], data["B365D"][i]]
            self.matches[name]=len(self.matches)

        self.data=DataSet(xData, yData)

    def nMatches(self):
        return len(self.matches)

    def nPlayers(self):
        return len(self.players)

    def getEmpiricalLS(self, lastN):

        len=self.nMatches()
        start=len-lastN
        temp=0
        for i in range(start, len):
            pH, pA, pD = self.getEmpirical(t=i)
            match=self.matches["match#"+str(i+1)]
            temp+=-(match.homeWin()*math.log(pH)+match.awayWin()*math.log(pA)+match.draw()*math.log(pD))
        return temp/lastN

    def getMeanLS(self, first=None, last=None, lastN=None):

        temp=0
        if lastN!=None:
            start=self.model.data.nMatches()-lastN
            end=self.model.data.nMatches()
        else:
            if first==None or last==None:
                raise Exception("define interval")
            start=first
            end=last


        for i in range(start, end):
            if self.game=="F":
                temp+=self.getOddsLS_F(self.matches["match#"+str(i+1)])
            elif self.game=="H":
                temp += self.getOddsLS_H(self.matches["match#" + str(i + 1)])
        return temp/(end-start)

    def getOddsLS_H(self, match):


        #print(match.odds)
        odds = [1 / float(match.odds[0]), 1 / float(match.odds[1])]
        bias = odds[0] + odds[1]
        r=1/bias
        print(r)

        homeWin=1
        if match.goalH<match.goalA:
            homeWin=0

        return -( (homeWin)*math.log(odds[0]*r) + (1-homeWin)*math.log(odds[1]*r) )

    def getOddsLS_F(self, match):

        #print(match.odds)
        odds = [1 / float(match.odds[0]), 1 / float(match.odds[1]), 1 / float(match.odds[2])]
        bias = odds[0] + odds[1] + odds[2]
        r=1/bias

        return -(match.homeWin()*math.log(odds[0]*r)+match.awayWin()*math.log(odds[1]*r)+match.draw()*math.log(odds[2]*r))

    def getMeanGains(self, mod, first=None, last=None, lastN=None):

        bet=1

        temp=0
        if lastN!=None:
            start=self.model.data.nMatches()-lastN
            end=self.model.data.nMatches()
        else:
            if first==None or last==None:
                raise Exception("define interval")
            start=first
            end=last

        for i in range(start, end):
            name="match#"+str(i+1)
            temp-=bet
            if self.game=="F":
                temp+=self.getGains_F(name, mod)*bet
            elif self.game=="H":
                temp+=self.getGains_H(name, mod)*bet

        return temp/(end-start)



    def getGains_H(self, matchName, mod):

        pH, pA, pD=mod.getProbs(matchName); pH+=pD/2; pA+=pD/2
        odds=self.matches[matchName].odds
        tomax=[pH*odds[0], pA*odds[1]]
        i = tomax.index(max(tomax))
        match=self.matches[matchName]
        bet=np.zeros(len(odds))
        bet[i]=odds[i]
        res=np.array(match.res)
        gains=np.dot(bet,res)
        return gains

    def getGains_F(self, matchName, mod):

        pH, pA, pD=mod.getProbs(matchName)
        odds=self.matches[matchName].odds
        tomax=[pH*odds[0], pA*odds[1], pD*odds[2]]
        i=tomax.index(max(tomax))
        match=self.matches[matchName]
        bet=np.zeros(len(odds))
        bet[i]=odds[i]
        res = np.array(match.res)
        gains = np.dot(bet,res)
        return gains


    def getEmpirical(self,start=None,end=None):

        if start==None:
            start=0
        if end==None:
            end=self.data.getSize()

        pees=np.zeros(3)

        for i in range(start,end):
            pees+=self.data.getY(i)

        return pees/(end-start)




