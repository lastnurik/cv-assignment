"""Derive figures directly from saved histories and evaluation artifacts."""
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/cv-matplotlib')
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from gesture_cv import CLASS_NAMES

OUT=Path('artifacts/evaluation')
NAMES={'B0':'B0 · Simple CNN','B1':'B1 · MobileNetV2','S1':'S1 · ConvNeXt pretrained','A0':'A0 · ConvNeXt random'}
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.titlesize':12,'axes.titleweight':'medium','axes.labelcolor':'#59676b','text.color':'#172d37','axes.edgecolor':'#d5d9d6','axes.spines.top':False,'axes.spines.right':False,'grid.color':'#d5d9d6','axes.prop_cycle':plt.cycler(color=['#315c48','#a34329','#54778c','#9a713a'])})
def main():
    comparison=json.loads((OUT/'comparison.json').read_text())
    fig,axes=plt.subplots(2,2,figsize=(10,7),layout='constrained')
    for ax,(role,result) in zip(axes.flat,comparison.items()):
        cm=result['overall']['confusion_matrix'];ax.imshow(cm,cmap='GnBu',vmin=0)
        for i,row in enumerate(cm):
            for j,value in enumerate(row):ax.text(j,i,str(value),ha='center',va='center',color='white' if value>12 else 'black')
        ax.set(xticks=range(3),yticks=range(3),xticklabels=['Peace','Stop','Thumbs up'],yticklabels=['Peace','Stop','Thumbs up'],title=NAMES[role]+' (n=51)',xlabel='Predicted',ylabel='True')
    fig.savefig(OUT/'confusion_matrices.png',dpi=220);plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(10,7),layout='constrained')
    for ax,role in zip(axes.flat,['B0','B1','S1','A0']):
        run=Path('runs')/(role+'_limited_v1');offset=0
        for phase in ['baseline','frozen','finetune']:
            p=run/(phase+'_history.json')
            if not p.exists():continue
            h=json.loads(p.read_text());epochs=list(range(offset+1,offset+len(h['loss'])+1))
            ax.plot(epochs,h['accuracy'],label=phase+' train');ax.plot(epochs,h['val_accuracy'],'--',label=phase+' val');offset+=len(epochs)
        ax.set(title=NAMES[role],xlabel='Epoch',ylabel='Accuracy',ylim=(0,1));ax.legend(fontsize=8);ax.grid(alpha=.2)
    fig.savefig(OUT/'training_curves.png',dpi=220);plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(10,7),layout='constrained')
    for ax,role in zip(axes.flat,['B0','B1','S1','A0']):
        offset=0
        for phase in ['baseline','frozen','finetune']:
            p=Path('runs')/(role+'_limited_v1')/(phase+'_history.json')
            if not p.exists():continue
            h=json.loads(p.read_text());epochs=list(range(offset+1,offset+len(h['loss'])+1));offset+=len(epochs)
            ax.plot(epochs,h['loss'],label=phase+' train');ax.plot(epochs,h['val_loss'],'--',label=phase+' val')
        ax.set(title=NAMES[role],xlabel='Epoch',ylabel='Cross-entropy');ax.legend(fontsize=8);ax.grid(alpha=.2)
    fig.savefig(OUT/'loss_curves.png',dpi=220);plt.close(fig)
if __name__=='__main__':main()
