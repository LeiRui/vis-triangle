from myfuncs import *
import os
import sys
import gc
from datetime import datetime

if len(sys.argv) != 2:
    print("Usage: python script.py <input_directory>")
    sys.exit(1)
    
mydir = sys.argv[1]
datasetNames = os.listdir(mydir)
print(len(datasetNames))

current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
pngDir=f"/root/tmpUCR_{current_time}"
if not os.path.exists(pngDir):
    os.makedirs(pngDir)

segmentDir=mydir
pattern=r"{dataset}-segment-{method}.csv"
    
anti=False
lw=0.7
width=1000
height=250
dpi=72
                
out="UCR_bench-large.csv"
with open(out, 'w', newline='') as outfile:
    writer = csv.writer(outfile)
    row=['classifier_name', 'dataset_name', 'accuracy'] # for CD diagram
    writer.writerow(row)
    
    for appendix in datasetNames:
        if '-segment' in appendix:
            continue

        filename=os.path.join(mydir,appendix)
        print(filename)
        df=pd.read_csv(filename, header=None)
        v=df.iloc[:,1]
        v=v.to_numpy(dtype='float')
        t=np.arange(len(v))

        nout=len(v)//2
        nout = (nout // 4) * 4 # for M4 requires at least integer multiply of four

        nout = min(800, nout)

        name=os.path.split(filename)[1].split(".")[0]+'_'+str(nout)+'_'+'original'
        downsample='original'
        subplt_myplot_external(width,height,dpi,name,anti,downsample,nout,lw,t,v,pngDir=pngDir,isPlot=False)
        originPNG=os.path.join(pngDir,name+'.png')

        ##############################################################
        name=os.path.split(filename)[1].split(".")[0]+'_'+str(nout)+'_'+'LTTBETDownsampler' 
        downsample='LTTBETDownsampler'
        t2,v2=subplt_myplot_external(width,height,dpi,name,anti,downsample,nout,lw,t,v,pngDir=pngDir,isPlot=False)
        f1=originPNG
        f2=os.path.join(pngDir,name+'.png')
        writer.writerow(['LTTB',filename,match(f1,f2)])
        gc.collect()

        ##############################################################
        name=os.path.split(filename)[1].split(".")[0]+'_'+str(nout)+'_'+'LTTBETFurtherDownsampler' 
        downsample='LTTBETFurtherDownsampler'
        t3,v3=subplt_myplot_external(width,height,dpi,name,anti,downsample,nout,lw,t,v,pngDir=pngDir,isPlot=False)
        f1=originPNG
        f2=os.path.join(pngDir,name+'.png')
        writer.writerow(['ILTS',filename,match(f1,f2)])
        gc.collect()

        ##############################################################
        name=os.path.split(filename)[1].split(".")[0]+'_'+str(nout)+'_'+'MinMaxLTTB2Downsampler'
        downsample='MinMaxLTTB2Downsampler'
        t4,v4,t4_pre,v4_pre=subplt_myplot_external(width,height,dpi,name,anti,downsample,nout,lw,t,v,\
                                                   pngDir=pngDir,isPlot=False)
        f1=originPNG
        f2=os.path.join(pngDir,name+'.png')
        writer.writerow(['MinMaxLTTB',filename,match(f1,f2)])
        gc.collect()

        ##############################################################
        name=os.path.split(filename)[1].split(".")[0]+'_'+str(nout)+'_'+'MinMaxFPLPDownsampler'
        downsample='MinMaxFPLPDownsampler'
        t5,v5=subplt_myplot_external(width,height,dpi,name,anti,downsample,nout,lw,t,v,pngDir=pngDir,isPlot=False)
        f1=originPNG
        f2=os.path.join(pngDir,name+'.png')
        writer.writerow(['MinMax',filename,match(f1,f2)])
        gc.collect()

        ##############################################################
        name=os.path.split(filename)[1].split(".")[0]+'_'+str(nout)+'_'+'M4Downsampler'
        downsample='M4Downsampler'
        t6,v6=subplt_myplot_external(width,height,dpi,name,anti,downsample,nout,lw,t,v,pngDir=pngDir,isPlot=False,\
                                    transform=True)
        f2=os.path.join(pngDir,name+'.png')

        name=os.path.split(filename)[1].split(".")[0]+'_'+str(nout)+'_'+'original_floored'
        downsample='original'
        subplt_myplot_external(width,height,dpi,name,anti,downsample,nout,lw,t,v,pngDir=pngDir,isPlot=False,\
                               transform=True)
        f1=os.path.join(pngDir,name+'.png')

        writer.writerow(['M4',filename,match(f1,f2)])
        gc.collect()

        ##############################################################
        name=os.path.split(filename)[1].split(".")[0]+'_'+str(nout)+'_'+'uniformTime'
        downsample='uniformTime'
        t10,v10=subplt_myplot_external(width,height,dpi,name,anti,downsample,nout,lw,t,v,pngDir=pngDir,isPlot=False)
        f1=originPNG
        f2=os.path.join(pngDir,name+'.png')
        writer.writerow(['Uniform',filename,match(f1,f2)])
        gc.collect()

        #############################################################
        resultFile=os.path.join(segmentDir,pattern.format(dataset=os.path.split(filename)[1].split(".")[0],\
                                          method="fsw"))
        print(resultFile)
        name=os.path.split(filename)[1].split(".")[0]+'_'+str(nout)+'_'+'FSW'
        downsample='FSW'
        t11,v11=subplt_myplot_external(width,height,dpi,name,anti,downsample,nout,lw,t,v,resultFile=resultFile,\
                                      pngDir=pngDir,isPlot=False)
        f1=originPNG
        f2=os.path.join(pngDir,name+'.png')
        writer.writerow(['FSW',filename,match(f1,f2)])
        gc.collect()

        #############################################################
        resultFile=os.path.join(segmentDir,pattern.format(dataset=os.path.split(filename)[1].split(".")[0],\
                                          method="simpiece"))
        print(resultFile)
        name=os.path.split(filename)[1].split(".")[0]+'_'+str(nout)+'_'+'SimPiece'
        downsample='SimPiece'
        t12,v12=subplt_myplot_external(width,height,dpi,name,anti,downsample,nout,lw,t,v,resultFile=resultFile,\
                                      pngDir=pngDir,isPlot=False)
        f1=originPNG
        f2=os.path.join(pngDir,name+'.png')
        writer.writerow(['Sim-Piece',filename,match(f1,f2)])
        gc.collect()


        ##############################################################
        area=getFastVisvalParam(nout,t,v,epsilon=1e-2)
        name=os.path.split(filename)[1].split(".")[0]+'_'+str(nout)+'_'+'fastVisval'
        downsample='fastVisval'
        t8,v8=subplt_myplot_external(width,height,dpi,name,anti,downsample,area,lw,t,v,pngDir=pngDir,isPlot=False)
        f1=originPNG
        f2=os.path.join(pngDir,name+'.png')
        writer.writerow(['Visval',filename,match(f1,f2)])
        print(len(t8))
        gc.collect()

        ################################################
        name=os.path.split(filename)[1].split(".")[0]+'_'+str(nout)+'_'+'DFT'
        downsample='PyDFT'
        t13,v13=subplt_myplot_external(width,height,dpi,name,anti,downsample,nout,lw,t,v,pngDir=pngDir,isPlot=False)
        f1=originPNG
        f2=os.path.join(pngDir,name+'.png')
        writer.writerow(['DFT',filename,match(f1,f2)])
        gc.collect()

        ################################################
        name=os.path.split(filename)[1].split(".")[0]+'_'+str(nout)+'_'+'PCA_auto'
        downsample='PCA_auto'
        t14,v14=subplt_myplot_external(width,height,dpi,name,anti,downsample,nout,lw,t,v,pngDir=pngDir,isPlot=False)
        f1=originPNG
        f2=os.path.join(pngDir,name+'.png')
        writer.writerow(['PCA',filename,match(f1,f2)])
        gc.collect()

        outfile.flush()
        
import shutil
if os.path.exists(pngDir):
    shutil.rmtree(pngDir)
    
print('finish')

################################################
from cd_diagram_Wilcoxon_Holm import *

df_perf = pd.read_csv(out) # , index_col=False
# Rename the columns to the desired names, because they are hard coded in wilcoxon_holm functions
df_perf.columns = ['classifier_name', 'dataset_name', 'accuracy']
print(df_perf['dataset_name'].nunique()/2)

# e.g.,
# classifier_name,dataset_name,accuracy
# clf3,dataset1,0.8197802197802198
# clf3,dataset2,0.80306905370844
# clf5,dataset1,0.4549450549450549
# clf5,dataset2,0.8388746803069054
# clf1,dataset1,0.6901098901098901
# clf1,dataset2,0.017902813299232736
# ...

print(df_perf)
draw_cd_diagram(df_perf=df_perf,colors=colorsmap,outPath="benchUCR")