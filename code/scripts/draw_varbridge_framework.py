"""Native editable scientific schematics. No raster inputs, model edits, data fits or training."""
from pathlib import Path
import hashlib,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch,FancyArrowPatch,Circle,Ellipse
from matplotlib.path import Path as MPath
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'test_results/varbridge_timebase_analysis/paper_diagrams';OUT.mkdir(exist_ok=True)
INK='#31465B';C={'blue':'#E7EFF7','teal':'#E3F1EC','peach':'#FBEDD9','purple':'#EDE8F5','coral':'#F7E4DF','white':'#FFFFFF'}
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':14,'svg.fonttype':'none','mathtext.fontset':'dejavusans','axes.linewidth':1})
def canvas(w,h,size):
 fig,ax=plt.subplots(figsize=size);ax.set(xlim=(0,w),ylim=(0,h));ax.axis('off');fig.subplots_adjust(0.015,0.015,.985,.985);return fig,ax

def box(ax,x,y,w,h,label,fill='white',fs=13,bold=False):
 ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.015,rounding_size=.13',fc=C.get(fill,fill),ec=INK,lw=1.2,zorder=2))
 t=ax.text(x+w/2,y+h/2,label,ha='center',va='center',fontsize=fs,fontweight='bold' if bold else 'normal',color=INK,zorder=3,linespacing=1.10)
 if not hasattr(ax,'_diagram_boxes'):ax._diagram_boxes=[]
 ax._diagram_boxes.append((t,x,y,w,h))
 return (x,y,w,h)
def txt(ax,x,y,s,fs=11,ha='center',weight='normal',color=INK):ax.text(x,y,s,ha=ha,va='center',fontsize=fs,color=color,fontweight=weight,zorder=5)
def line(ax,points,dashed=False,color=INK,lw=1.3,arrow=True):
 p=MPath(points,[MPath.MOVETO]+[MPath.LINETO]*(len(points)-1))
 ax.add_patch(FancyArrowPatch(path=p,arrowstyle='-|>' if arrow else '-',mutation_scale=12,lw=lw,color=color,linestyle=(0,(4,3)) if dashed else '-',zorder=1,joinstyle='round'))
def plus(ax,x,y):
 ax.add_patch(Circle((x,y),.24,fc='white',ec=INK,lw=1.2,zorder=3));txt(ax,x,y,'+',16)
def frame(ax,x,y,w,h,title):
 ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.02,rounding_size=.15',fc='#FAFCFD',ec='#A4B3BF',lw=1,zorder=0));txt(ax,x+.25,y+h-.30,title,14,ha='left',weight='bold')
def save(fig,name):
 fig.canvas.draw();renderer=fig.canvas.get_renderer()
 for ax in fig.axes:
  for text,x,y,w,h in getattr(ax,'_diagram_boxes',[]):
   lo=ax.transData.transform((x,y));hi=ax.transData.transform((x+w,y+h));bb=text.get_window_extent(renderer)
   factor=min(1.,.91*(hi[0]-lo[0])/max(bb.width,1),.85*(hi[1]-lo[1])/max(bb.height,1))
   if factor<1:text.set_fontsize(text.get_fontsize()*factor)
 fig.canvas.draw()
 for ax in fig.axes:
  for text,x,y,w,h in getattr(ax,'_diagram_boxes',[]):
   lo=ax.transData.transform((x,y));hi=ax.transData.transform((x+w,y+h));bb=text.get_window_extent(fig.canvas.get_renderer())
   assert bb.x0>=lo[0]-1 and bb.x1<=hi[0]+1 and bb.y0>=lo[1]-1 and bb.y1<=hi[1]+1,(name,text.get_text())
 fig.savefig(OUT/(name+'.svg'),facecolor='white')
 fig.savefig(OUT/(name+'.png'),dpi=600,facecolor='white');plt.close(fig)

# Figure1: the sources and distinct text roles. Curves are explicit schematics, not measurements.
fig,ax=canvas(20,10.3,(11,5.65))
box(ax,.25,7.1,6.1,2.65,'','blue');txt(ax,3.3,9.35,'Multi-sensor history',16,weight='bold')
# Minimal reactor illustration, drawn from primitives.
ax.add_patch(Ellipse((1.15,8.55),1.2,.4,fill=False,ec=INK,lw=1.3,zorder=3));ax.plot([.55,.55,1.75,1.75],[8.55,7.55,7.55,8.55],color=INK,lw=1.3)
for x in [.85,1.15,1.45]:ax.plot([x,x],[8.95,8.05],color=INK,lw=1);ax.plot([x-.09,x+.09],[8.95,8.95],color=INK,lw=1.4)
ax.plot([.65,1.65],[7.9,7.9],color=INK,lw=1);ax.plot([1.15,1.15],[8.7,7.65],color=INK,lw=1)
t=np.linspace(0,1,160)
for j,c in enumerate(['#4B83B4','#438E79','#8E71AC']):ax.plot(2.1+3.9*t,8.85-.55*j+.11*np.sin((3+j)*2*np.pi*t+j),color=c,lw=1.65)
box(ax,8.1,7.45,4.3,1.7,'Numerical\nrepresentations','teal',16,True)
box(ax,14.25,7.1,5.5,2.65,'','coral');txt(ax,17,9.35,'Future trajectories',15,weight='bold')
for j,c in enumerate(['#4B83B4','#438E79','#8E71AC']):
 q=8.85-.55*j+.11*np.sin((3+j)*2*np.pi*t+j);ax.plot(14.7+4.6*t[t<=.57],q[t<=.57],color=c,lw=1.6);ax.plot(14.7+4.6*t[t>.57],q[t>.57],color=c,lw=1.6,ls='--')
ax.plot([17.32,17.32],[7.5,9.0],color=INK,lw=1,ls='--')
line(ax,[(6.35,8.3),(8.1,8.3)]);line(ax,[(12.4,8.3),(14.25,8.3)])
for y,label in [(5.7,'Observed-window summary'),(4.75,'Stage context'),(3.8,'Operating context')]:box(ax,.25,y,5.7,.72,label,'peach',13)
box(ax,7.9,4.4,4.45,1.1,'General process\ncontext','peach',14,True)
for y in [6.06,5.11,4.16]:line(ax,[(5.95,y),(6.8,y),(6.8,4.95),(7.9,4.95)])
line(ax,[(10.1,5.5),(10.1,7.45)])
for y,name,meaning in [(2.55,'Biomass','Growth and substrate use'),(1.65,'Dissolved oxygen','Transfer and demand'),(.75,'pH','Acid-base environment')]:
 box(ax,.25,y,6.6,.7,name+'  |  '+meaning,'purple',12.5)
box(ax,9.1,1.2,4.0,1.65,'Variable-bound\nconditioning\n(per target)','purple',14,True)
for y in [2.9,2.0,1.1]:line(ax,[(6.85,y),(7.75,y),(7.75,2.02),(9.1,2.02)])
line(ax,[(12.4,7.85),(13.5,7.85),(13.5,3.4),(11.2,3.4),(11.2,2.85)],color='#438E79')
txt(ax,14.05,5.65,'Historical\nstate',12,ha='left')
box(ax,8.9,.10,4.4,.65,'Forecast step','blue',12);line(ax,[(11.1,.75),(11.1,1.2)])
line(ax,[(13.1,2.02),(18.5,2.02),(18.5,7.1)]);txt(ax,16.5,3.7,'Target-specific\nforecast conditions',12)
save(fig,'varbridge_process_concept')

# Figure2: exact variable-bound condition, including the one-token projection equivalence.
fig,ax=canvas(24,9,(12,4.8))
box(ax,.2,6.9,3.0,1.3,'Target description\n'+r'$m_j$','purple',14)
box(ax,3.75,6.9,3.0,1.3,'Frozen\nRoBERTa','blue',14)
box(ax,7.3,6.9,3.0,1.3,'Description\nprojection','purple',14)
plus(ax,11.4,7.55);box(ax,12.25,6.9,2.75,1.3,'Bound vector\n'+r'$\mathbf{k}_j$','purple',14)
box(ax,15.55,6.9,3.1,1.3,'Single-token\nprojection','blue',14)
box(ax,19.2,6.9,4.5,1.3,'Mechanism context\n'+r'$\mathbf{c}_j$','purple',14)
for a,b in [(3.2,3.75),(6.75,7.3),(10.3,11.16),(11.64,12.25),(15,15.55),(18.65,19.2)]:line(ax,[(a,7.55),(b,7.55)])
txt(ax,21.45,8.62,'One context per target',12)
box(ax,7.1,4.85,3.45,1.0,'Variable identity\n'+r'$\mathbf{u}_j$','blue',13)
line(ax,[(10.55,5.35),(11.4,5.35),(11.4,7.31)])
box(ax,.2,3.1,3,1.3,'Historical state\n'+r'$\mathbf{a}_j$','teal',14)
box(ax,3.75,3.1,3,1.3,'State\nprojection','teal',14);plus(ax,8.8,3.75)
box(ax,10,3.1,4.2,1.3,'Conditional query\n'+r'$\mathbf{q}_{j,h}$','teal',14)
box(ax,17,3.1,6.7,1.3,'Concatenate\n'+r'$\mathbf{f}_{j,h}=[\mathbf{q}_{j,h};\mathbf{c}_j]$','peach',15)
line(ax,[(3.2,3.75),(3.75,3.75)]);line(ax,[(6.75,3.75),(8.56,3.75)]);line(ax,[(9.04,3.75),(10,3.75)]);line(ax,[(14.2,3.75),(17,3.75)])
line(ax,[(8.8,4.85),(8.8,3.99)])
box(ax,7.05,.7,3.5,1.15,'Forecast-step\nembedding '+r'$\mathbf{v}_h$','blue',13);line(ax,[(8.8,1.85),(8.8,3.51)])
line(ax,[(21.45,6.9),(21.45,4.4)])
for x,label in [(13.4,'Numerical\nrouter'),(17,'Correction\nrouter'),(20.6,'Final forecast\nrouter')]:
 box(ax,x,.7,3.1,1.25,label,'coral',13);line(ax,[(20.35,3.1),(20.35,2.55),(x+1.55,2.55),(x+1.55,1.95)])
txt(ax,18.5,.22,'Shared condition; separate router parameters',12)
save(fig,'varbridge_variable_bound_condition')

# Figure3: functional graph with explicit connector tags, avoiding ambiguous shared wires.
fig,ax=canvas(24,16,(9.6,6.4))
frame(ax,.2,10.6,23.6,5.1,'(a) Numerical and text representations')
box(ax,.5,13.0,2.5,1.3,'Observed\nhistory '+r'$\mathbf{X}$','blue',13)
box(ax,3.5,13.0,2.8,1.3,'Window\nnormalization','blue',13)
box(ax,6.8,13.0,2.8,1.3,'Frozen\nPatchTST','blue',14)
box(ax,10.2,13.0,3.2,1.3,'Early variable\nattention','teal',13)
line(ax,[(3,13.65),(3.5,13.65)]);line(ax,[(6.3,13.65),(6.8,13.65)]);line(ax,[(9.6,13.65),(10.2,13.65)]);txt(ax,9.9,14.07,r'$\mathbf{F}$',12)
box(ax,14.2,10.95,6.0,3.4,'','teal');txt(ax,17.2,14.03,'Numerical adaptation',13,weight='bold')
for y,label in [(13.18,'EMA trend and residual patches'),(12.58,'Dynamic window statistics'),(11.98,'Process-text cross-attention'),(11.38,'Late variable attention')]:box(ax,14.45,y,5.5,.43,label,'white',10.8)
line(ax,[(12.85,13.65),(14.2,13.65)])
# Two source-specific inputs, deliberately separated from the backbone bypass tag.
line(ax,[(1.75,13),(1.75,11.5),(14.2,11.5)]);txt(ax,6.8,11.76,'Statistics from X',10.5)
line(ax,[(4.9,13),(4.9,10.82),(13.9,10.82),(13.9,11.05),(14.2,11.05)]);txt(ax,9.1,10.98,'Normalized history',10.5)
box(ax,9.15,11.9,3.1,.65,'Base forecast '+r'$\mathbf{b}$','blue',12);line(ax,[(8.2,13),(8.2,12.225),(9.15,12.225)])
box(ax,13.7,14.85,4.2,.7,'General process text','peach',10.5)
box(ax,18.65,14.85,3.25,.7,'Frozen RoBERTa','blue',12)
line(ax,[(17.9,15.2),(18.65,15.2)]);line(ax,[(20.275,14.85),(20.275,14.55),(18.4,14.55),(18.4,14.35)])
txt(ax,22.75,14.95,'General\ncontext',10.5)
box(ax,21.15,12.25,2.4,1.75,'Trend U +\nadapted\nresidual','teal',12)
line(ax,[(20.2,13.12),(21.15,13.12)])
frame(ax,.2,7.0,23.6,3.35,'(b) Variable-bound forecast conditions')
box(ax,.5,7.75,3.0,1.1,'Own-variable\ndescription','purple',13)
box(ax,4.0,7.75,2.7,1.1,'Frozen\nRoBERTa','blue',13)
box(ax,7.2,7.75,3.1,1.1,'Variable-bound\nprojection','purple',12)
box(ax,10.8,7.75,3.1,1.1,'Mechanism\ncontext c','purple',13)
for a,b in [(3.5,4.0),(6.7,7.2),(10.3,10.8)]:line(ax,[(a,8.3),(b,8.3)])
box(ax,17.7,7.45,5.6,1.45,'Joint forecast\ncondition f','peach',15,True)
line(ax,[(13.9,8.3),(14.35,8.3),(14.35,7.25),(18.35,7.25),(18.35,7.45)])
box(ax,14.8,9.12,5.3,.65,'State: early F + trend U','teal',10.4)
line(ax,[(20.1,9.445),(20.5,9.445),(20.5,8.9)])
box(ax,14.8,8.24,2.5,.65,'Variable ID','blue',11);line(ax,[(17.3,8.565),(17.7,8.565)])
box(ax,14.8,7.35,2.5,.65,'Forecast step','blue',11);line(ax,[(17.3,7.675),(17.7,7.675)])
# The projection's identity dependence is expanded in Figure2, rather than hidden in a crossing wire.
txt(ax,5.5,7.23,'Binding and single-token projection are expanded in Fig. 2.',10.2)
frame(ax,.2,.02,23.6,6.7,'(c) Hierarchical forecast fusion')
box(ax,9.4,5.96,4.2,.45,'Joint forecast condition f','peach',11)
for x in [11.5,17.5,22.0]:line(ax,[(11.5,5.96),(11.5,5.77),(x,5.77),(x,5.65)],dashed=True)
box(ax,3.4,4.65,4.5,1.0,'Trend U +\nadapted residual','teal',12)
box(ax,9.4,4.65,4.2,1.0,'Numerical experts\n+ router','teal',13)
box(ax,15.4,4.65,4.2,1.0,'Correction experts\n+ router','purple',13)
box(ax,20.7,4.65,2.6,1.0,'Final\nrouter','teal',13)
line(ax,[(7.9,5.15),(9.4,5.15)])
box(ax,7.5,3.65,1.65,.55,'Step '+r'$\mathbf{v}_h$','blue',10.5);line(ax,[(9.15,3.925),(9.27,3.925),(9.27,4.9),(9.4,4.9)])
box(ax,3.8,3.65,3.4,.6,'Base b','blue',12)
box(ax,19.7,3.45,2.0,.65,'Base b','blue',12)
plus(ax,17.5,3.8);line(ax,[(17.5,4.65),(17.5,4.04)]);txt(ax,18.35,4.2,r'$\Delta$',12);line(ax,[(19.7,3.775),(17.74,3.775)])
for x,label in [(3.25,'Base forecast b'),(9.25,'Numerical\nforecast n'),(15.25,'Corrected base\nb + Δ')]:box(ax,x,2.5,4.5,.85,label,'coral',13)
line(ax,[(5.5,3.65),(5.5,3.35)]);line(ax,[(11.5,4.65),(11.5,3.35)]);line(ax,[(17.5,3.56),(17.5,3.35)])
box(ax,6.0,1.2,12.0,.65,'Weighted forecast fusion','coral',15,True)
line(ax,[(5.5,2.5),(5.5,2.1),(7,2.1),(7,1.85)]);line(ax,[(11.5,2.5),(11.5,1.85)]);line(ax,[(17.5,2.5),(17.5,1.85)])
line(ax,[(22,4.65),(22,1.525),(18,1.525)],dashed=True)
box(ax,7.0,.17,6.0,.65,'Inverse window norm.','blue',12)
box(ax,14.5,.17,5.0,.65,'Future trajectories','coral',13,True)
line(ax,[(12,1.2),(12,1.0),(10,.82)]);line(ax,[(13,.495),(14.5,.495)])
save(fig,'varbridge_full_architecture')
(OUT/'diagram_provenance.json').write_text(json.dumps({'kind':'native geometric vector schematics; no generated raster input embedded','purpose':'concept / exact variable-bound conditioning / full functional architecture','source':'current adopted VarBridge manuscript; original model unchanged','palette':C,'schematic_waveforms':True,'experimental_results_modified':False,'generator_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},indent=2))
print('THREE_NATIVE_DIAGRAMS_COMPLETE',OUT)
