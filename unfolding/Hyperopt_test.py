#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
import ROOT 
from hyperopt import fmin, tpe, hp,Trials
import scipy.stats as stats
from array import array

## Helper functions 

def MakeListResponse(h2):
    vals = []
    for i in range(1, h2.GetYaxis().GetNbins()+1):
        column = []
        for j in range(1, h2.GetXaxis().GetNbins()+1):
            val = h2.GetBinContent(i,j)
            column.append(val)
        vals.append(column)
    return vals

def NormalizeResponse(h2, tag = '_migra'):
    migra = h2.Clone(h2.GetName() + tag)
    for i in range(1, h2.GetXaxis().GetNbins()+1):
        sum = 0.
        for j in range(1, h2.GetYaxis().GetNbins()+1):
            val = h2.GetBinContent(i,j)
            sum = sum + val
        if sum > 0.:
            for j in range(1, h2.GetYaxis().GetNbins()+1):
                migra.SetBinContent(i,j,migra.GetBinContent(i,j) / sum)
    return migra

def TransposeMatrix(h_response_unf):
    h_responce_transpose = h_response_unf.Clone(h_response_unf.GetName()+"clone")
    h_responce_transpose.Reset()
    for i in range(1,h_response_unf.GetXaxis().GetNbins()+1):
        for j in range(1,h_response_unf.GetXaxis().GetNbins()+1):
            h_responce_transpose.SetBinContent(i,j,h_response_unf.GetBinContent(j,i))
            h_responce_transpose.SetBinError(i,j,h_response_unf.GetBinError(j,i))
    TitleX = h_response_unf.GetXaxis().GetTitle()
    TitleY = h_response_unf.GetYaxis().GetTitle()
    h_responce_transpose.GetXaxis().SetTitle(TitleY)
    h_responce_transpose.GetYaxis().SetTitle(TitleX)
    return h_responce_transpose

def MakeListFromHisto(hist):
    vals = []
    for i in range(1, hist.GetXaxis().GetNbins()+1):
        val = hist.GetBinContent(i)
        vals.append(val)
    return vals

def MyClone(hist,MyName):
    h = hist.Clone(hist.GetName()+"_myclone"+str(MyName))
    h.Reset("M")
    for i in range(1, hist.GetXaxis().GetNbins()+1):
        h.SetBinContent(i,hist.GetBinContent(i))
        h.SetBinError(i,hist.GetBinError(i))
    TitleX = hist.GetXaxis().GetTitle()
    TitleY = hist.GetYaxis().GetTitle()
    h.GetXaxis().SetTitle(TitleY)
    h.GetYaxis().SetTitle(TitleX)
    return h

## prepare root
InputFile = ROOT.TFile("input.root", 'read')
#Data = InputFile.Get("Reco/HadTopPt")
#Data = InputFile.Get("Particle/MCHadTopPt")
Matrix = InputFile.Get("Matrix/MigraHadTopPt")
Truth = InputFile.Get("Particle/MCHadTopPt")
Data = MyClone(Truth, "_data")
NRebin = 100
Data.Rebin(NRebin)
Matrix.Rebin2D(NRebin,NRebin)
Truth.Rebin(NRebin)

can = ROOT.TCanvas("can","can",0,0,1600,1600)
can.Divide(2,2)
can.cd(1)

########### ACCEPTANCY
h_acc = Matrix.ProjectionX("reco_recoandparticleX") # Reco M
h_acc.Divide(Truth)
########### AKCEPTANCE saved in h_acc #############
########### EFFICIENCY
h_eff = Matrix.ProjectionY("reco_recoandparticleY") # Ptcl M
h_eff.Divide(Truth)

h_acc.SetLineColor(1)
h_acc.Draw("hist")
h_eff.SetLineColor(4)
h_eff.Draw("hist same")

#DataOr = InputFile.Get("Reco/HadTopPt")
#DataOr = InputFile.Get("Particle/MCHadTopPt")
DataOr = MyClone(Data, "data_or")
DataOr.SetLineColor(1)
can.cd(2)
DataOr.Draw("hist")
Truth.Draw("hist same")
can.cd(3)
Matrix.Draw("colz text")

Data.Multiply(h_acc)


Unfolded = MyClone(Truth, "Unfoldeld")
Unfolded.SetName("Unfolded")
#Unfolded.Reset()
Dim = Data.GetXaxis().GetNbins()

#Matrix.Draw("colz")
Matrix.ClearUnderflowAndOverflow()
Matrix.GetXaxis().SetRange(1, Matrix.GetXaxis().GetNbins() )
Matrix.GetYaxis().SetRange(1, Matrix.GetYaxis().GetNbins() )
Matrix.SetName("Migration_Matrix_simulation")
#Matrix = TransposeMatrix(Matrix)
Matrix2 = NormalizeResponse(Matrix)
can.cd(4)
Matrix2.Draw("colz text")
Matrix = MakeListResponse(Matrix2)
Matrix_np = np.array(Matrix)
Data_np = np.array(MakeListFromHisto(Data))
Reco_temp_np = np.array(MakeListFromHisto(Data))

#Matrix_np = Matrix_np.astype(np.int64)
#Data_np = Data_np.astype(np.int64)
#Reco_temp_np = Reco_temp_np.astype(np.int64)



number_of_experiments = 1000
#Define the Rosenbrock function as the objective
def rosenbrock_objective(args):
    #x = args['x']
    #y = args['y']
    Reco_temp_np = []
    for i in range(len(args)):
        Reco_temp_np.append(args[str(i+1)])
    sum = 0.0
    Reco_temp_np = np.array(Reco_temp_np)
    Reco_temp_np = Reco_temp_np.dot(Matrix_np)
    for i in range(len(args)):
        sum+=Data_np[i]*np.log(Reco_temp_np[i]) - Reco_temp_np[i] - ROOT.Math.lgamma(Data_np[i]+1.0)
    #print(-1.0*sum)
    return -1.0*sum

#Trials keeps track of all experiments
#These can be saved and loaded buack into a new batch of experiments
trials_to_keep = Trials()
#Space is where you define the parameters and parameter search space
space = {}
for i in range(1, Dim+1):
    space[str(i)] = hp.uniform(str(i), 0.0, 10*Data.GetBinContent(i))
    #space[str(i)] = hp.uniform(str(i), Data.GetBinContent(i)/10.0, 5*Data.GetBinContent(i))
#space={'x': hp.uniform('x', -2, 2),'y': hp.uniform('y', -2, 2)}
print(space)
#The main function to run the experiments
#The algorithm tpe.suggest runs the Tree-structured Parzen estimator
#Hyperopt does have a random search algorithm as well
best = fmin(fn=rosenbrock_objective, space=space, algo=tpe.suggest, max_evals=number_of_experiments, trials = trials_to_keep )
print(best)
for i in range(len(best)):
    Unfolded.SetBinContent(i+1, best[str(i+1)])
Unfolded.Divide(h_eff)
Unfolded.SetLineColor(2)
can.cd(2)
Unfolded.Draw("hist same")
Truth.SetLineColor(4)
Truth.Draw("Hist same")
xvals = []
yvals = []
zvals = []
for trial in trials_to_keep.trials:
    #print("x = ",trial['misc']['vals']['x'][0])
    #print("y = ",trial['misc']['vals']['y'][0])
    #print("z = ",trial['result']['loss'])
    xvals.append(trial['misc']['vals']['3']) #3ty bin vs 4ty bin
    yvals.append(trial['misc']['vals']['4'])
    zvals.append(-1.0*trial['result']['loss'])

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(xvals,yvals,zvals, c=zvals,cmap='rainbow', s=50, alpha=0.8)
plt.show()
#{'state': 2, 'tid': 99, 'spec': None, 'result': {'loss': 0.8656630378655292, 'status': 'ok'}, 'misc': {'tid': 99, 'cmd': ('domain_attachment', 'FMinIter_Domain'), 'workdir': None, 'idxs': {'x': [99], 'y': [99]}, 'vals': {'x': [0.1284859468749301], 'y': [2.3257703072274114]}}, 'exp_key': None, 'owner': None, 'version': 0, 'book_time': datetime.datetime(2020, 11, 4, 7, 13, 35, 755000), 'refresh_time': datetime.datetime(2020, 11, 4, 7, 13, 35, 755000)}


