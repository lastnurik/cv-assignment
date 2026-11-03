"""Create the defense PPTX through installed LibreOffice UNO.

Run with /usr/bin/python3 while an isolated headless UNO listener uses port 2011.
All tables and learning curves remain native editable PowerPoint objects.
"""
import json
from pathlib import Path
import uno
from com.sun.star.awt import Point,Size
from com.sun.star.beans import PropertyValue
from com.sun.star.drawing.FillStyle import SOLID,NONE
from com.sun.star.drawing.LineStyle import NONE as NO_LINE
from com.sun.star.style.ParagraphAdjust import LEFT
ROOT=Path(__file__).resolve().parents[1]
BUILD=ROOT/'artifacts/qa/defense_pptx'
OUT=ROOT/'deliverables/A2_Defense_Presentation.pptx'
PX=2540/96
INK=0x172D37;MUTED=0x59676B;PAPER=0xF5F4EF;ACCENT=0xA34329;GREEN=0x315C48
def prop(k,v):
 p=PropertyValue();p.Name=k;p.Value=v;return p
def read(p):return json.loads((ROOT/p).read_text())
def url(p):return uno.systemPathToFileUrl(str(Path(p).resolve()))
def main():
 local=uno.getComponentContext()
 resolver=local.ServiceManager.createInstanceWithContext('com.sun.star.bridge.UnoUrlResolver',local)
 ctx=resolver.resolve('uno:socket,host=127.0.0.1,port=2011;urp;StarOffice.ComponentContext')
 sm=ctx.ServiceManager;desktop=sm.createInstanceWithContext('com.sun.star.frame.Desktop',ctx)
 doc=desktop.loadComponentFromURL('private:factory/simpress','_blank',0,(prop('Hidden',True),))
 pages=doc.getDrawPages();evidence=[]
 def box(slide,x,y,w,h,text,size=26,color=INK,font='Arial',bold=False):
  sh=doc.createInstance('com.sun.star.drawing.TextShape');sh.Position=Point(round(x*PX),round(y*PX));sh.Size=Size(round(w*PX),round(h*PX));slide.add(sh)
  sh.String=text;sh.TextAutoGrowHeight=False;sh.TextAutoGrowWidth=False
  sh.TextLeftDistance=0;sh.TextRightDistance=0;sh.TextUpperDistance=0;sh.TextLowerDistance=0
  sh.FillStyle=NONE;sh.LineStyle=NO_LINE
  sh.CharFontName=font;sh.CharHeight=size*.75;sh.CharColor=color;sh.CharWeight=150 if bold else 100;sh.ParaAdjust=LEFT
  return sh
 def new(title,notes):
  idx=len(evidence)
  sl=pages.getByIndex(0) if idx==0 else pages.insertNewByIndex(idx)
  for i in range(sl.Count-1,-1,-1):sl.remove(sl.getByIndex(i))
  sl.Width=round(1280*PX);sl.Height=round(720*PX)
  bg=doc.createInstance('com.sun.star.drawing.Background');bg.FillStyle=SOLID;bg.FillColor=PAPER;sl.Background=bg
  if title:box(sl,64,42,1152,88,title,44,font='Georgia')
  box(sl,64,678,1000,22,'Gesture Lab / Computer Vision / Assignment 2',16,MUTED)
  box(sl,1176,678,42,22,str(idx+1),16,MUTED)
  for i in range(sl.NotesPage.Count):
   sh=sl.NotesPage.getByIndex(i)
   if sh.ShapeType=='com.sun.star.presentation.NotesShape':sh.String=notes
  evidence.append({'slide':idx+1,'title':title,'notes':notes})
  return sl
 def image(slide,path,x,y,w,h,alt):
  provider=sm.createInstanceWithContext('com.sun.star.graphic.GraphicProvider',ctx)
  graphic=provider.queryGraphic((prop('URL',url(ROOT/path)),))
  shape=doc.createInstance('com.sun.star.drawing.GraphicObjectShape');slide.add(shape)
  # Preserve source proportions using the graphic's pixel dimensions.
  sz=graphic.SizePixel;scale=min(w/sz.Width,h/sz.Height);iw=sz.Width*scale;ih=sz.Height*scale
  shape.Position=Point(round((x+(w-iw)/2)*PX),round((y+(h-ih)/2)*PX));shape.Size=Size(round(iw*PX),round(ih*PX))
  shape.Graphic=graphic;shape.Title=alt;shape.Description=alt
  return shape
 def table(slide,values,x,y,widths,rowheight=62):
  sh=doc.createInstance('com.sun.star.drawing.TableShape');slide.add(sh)
  model=sh.Model
  model.Columns.insertByIndex(0,len(widths)-1);model.Rows.insertByIndex(0,len(values)-1)
  for j,w in enumerate(widths):model.Columns.getByIndex(j).Width=round(w*PX)
  for i,row in enumerate(values):
   model.Rows.getByIndex(i).Height=round(rowheight*PX)
   for j,value in enumerate(row):
    c=model.getCellByPosition(j,i);c.String=str(value)
    c.CharFontName='Arial';c.CharHeight=18 if i==0 else 20;c.CharWeight=150 if i==0 else 100
    c.CharColor=0xffffff if i==0 else INK;c.FillColor=INK if i==0 else (0xE7ECE9 if i%2 else PAPER)
    c.TextLeftDistance=round(12*PX);c.TextRightDistance=round(8*PX);c.TextUpperDistance=round(8*PX);c.TextLowerDistance=round(8*PX)
  sh.Position=Point(round(x*PX),round(y*PX));sh.Size=Size(round(sum(widths)*PX),round(len(values)*rowheight*PX))
  return sh
 def curve(slide,role,x,y,w,h):
  hist=read(f'runs/{role}_limited_v1/frozen_history.json')
  shape=doc.createInstance('com.sun.star.drawing.OLE2Shape');shape.Position=Point(round(x*PX),round(y*PX));shape.Size=Size(round(w*PX),round(h*PX));shape.CLSID='12DCAE26-281F-416F-a234-c3086127382e';slide.add(shape)
  ch=shape.Model;ch.Diagram=ch.createInstance('com.sun.star.chart.LineDiagram')
  ch.Data.setData(tuple(zip(hist['accuracy'],hist['val_accuracy'])))
  ch.Data.setColumnDescriptions(('Training','Validation'));ch.Data.setRowDescriptions(tuple(str(i+1) for i in range(len(hist['accuracy']))))
  ch.HasMainTitle=True;ch.Title.String='S1: pretrained' if role=='S1' else 'A0: random features';ch.Title.CharFontName='Arial';ch.Title.CharHeight=19
  ch.HasLegend=True;ch.Legend.CharFontName='Arial';ch.Legend.CharHeight=14
  ch.Area.FillColor=PAPER;ch.Area.LineStyle=NO_LINE;ch.Diagram.Wall.FillColor=PAPER;ch.Diagram.Wall.LineStyle=NO_LINE
  diag=ch.Diagram;diag.HasXAxisTitle=True;diag.XAxisTitle.String='Epoch';diag.XAxisTitle.CharFontName='Arial';diag.XAxisTitle.CharHeight=14
  diag.HasYAxisTitle=True;diag.YAxisTitle.String='Accuracy';diag.YAxisTitle.CharFontName='Arial';diag.YAxisTitle.CharHeight=14
  for axis in [diag.XAxis,diag.YAxis]:axis.CharFontName='Arial';axis.CharHeight=13;axis.CharColor=MUTED
  diag.YAxis.AutoMin=False;diag.YAxis.Min=0;diag.YAxis.AutoMax=False;diag.YAxis.Max=1
  for i,col in enumerate([GREEN,ACCENT]):
   ser=diag.getDataRowProperties(i);ser.LineColor=col;ser.LineWidth=80
  return shape

 results=read('artifacts/evaluation/comparison.json')
 s=new('',"Opening: We continued the A1 hand gesture task and added ConvNeXtTiny, a working local app and Grad-CAM. Three supported classes: peace, stop and thumbs up. The same team dataset underlies every comparison. Course-use photo consent is confirmed; public redistribution has not been authorized.\nSource image: data/clean/a1_clean_v1/thumbs_up/thumbs_up_0064.png.")
 box(s,64,106,590,50,'Gesture Lab',30,ACCENT)
 box(s,64,184,620,205,'Hand gesture\nrecognition',72,font='Georgia')
 box(s,64,430,590,90,'ConvNeXt, baseline comparisons\nand visual explanations',29,MUTED)
 box(s,64,570,590,55,'Astana IT University\nAssignment 2 defense',22,MUTED)
 image(s,'data/clean/a1_clean_v1/thumbs_up/thumbs_up_0064.png',730,155,486,390,'Cleaned thumbs-up image from the existing dataset')

 s=new('The working app',"The local web app accepts an upload or camera capture, predicts one of three gestures and computes Grad-CAM. Demonstrate upload during defense because physical camera capture has not been verified. The interface handles denied camera access by offering upload. It never saves uploaded images. The crop is applied exactly once. Prediction probabilities are not passed through softmax twice.\nEvidence: artifacts/qa/browser_qa.json; artifacts/qa/refinement_v2/camera_state.json. Image: artifacts/qa/browser/app_workspace.png.")
 image(s,'artifacts/qa/browser/app_workspace.png',64,150,1152,430,'Actual app showing a peace prediction and Grad-CAM')
 box(s,64,602,1152,48,'Upload a photo, inspect the probabilities, then examine the heatmap.',26)
 
 s=new('Dataset and leakage controls',"A1 contains 464 original images. A fixed top-20% crop removes burned-in class labels. Visual review excluded 11 stop images without a visible gesture, leaving 453 images. Originals remain preserved. We froze the same split for all four models. Thirty-eight dHash64 similarity components, including mixed-class components, remain intact within splits. Known exact/source duplicates and similarity edges have zero cross-split overlap. These groups do not imply independent sessions: the original session IDs are unknown, and all images share one main participant and indoor room. Stop has only four test examples.\nSource: data/manifests/a1_limited_v2.json; artifacts/qa/frozen_data_validation.json; docs/dataset_card.md.")
 box(s,64,148,1152,70,'464 original images. Remove collection labels and exclude 11 no-hand images.',27)
 table(s,[['Class','Train','Validation','Test'],['Peace',105,23,23],['Stop',136,12,4],['Thumbs up',103,23,24],['Total',344,58,51]],64,242,[432,240,240,240],60)
 box(s,64,574,1152,64,'Similarity groups stay within a split. One person and one setting limit generalization.',25,MUTED)

 s=new('ConvNeXt and the pretraining ablation',"ConvNeXt modernizes convolutional networks using large depthwise kernels, an inverted bottleneck, LayerNorm and GELU. The architecture comes from CVPR 2022; we do not claim it is the best method in 2026. Tiny is suitable for a local CPU demonstration. The spatial backbone feeds global average pooling and a three-class Dense head. It receives RGB pixels in [0,255] because normalization is embedded. Spatial features remain available for Grad-CAM.\nA0 and S1 freeze the same backbone architecture and train identically initialized Dense heads. They share data, seed, batch size, optimizer and twenty-epoch training budget with no augmentation. Backbone initialization is the only ablated factor. B0/B1 are separate architecture comparisons and retain their A1 training procedures.\nSources: https://arxiv.org/abs/2201.03545 ; https://keras.io/api/applications/convnext/ ; configs/experiment_contract.convnext_v1.json.")
 box(s,64,146,1152,76,'A frozen ConvNeXtTiny backbone supplies features to a three-class head.',29)
 table(s,[['Control','S1','A0'],['Backbone initialization','ImageNet weights','Random weights'],['Backbone training','Frozen','Frozen'],['Head initialization','Same seed 223','Same seed 223'],['Head training','20 epochs, Adam 0.001','20 epochs, Adam 0.001']],64,252,[432,360,360],60)
 box(s,64,586,1152,44,'ConvNeXt: Liu et al., CVPR 2022. No synthetic training images.',23,MUTED)

 s=new('Training and validation behavior',"These are native editable charts built from the saved 20-epoch histories. Pretrained ConvNeXt reaches high validation accuracy quickly. Random frozen features allow the head to fit the training set but leave a large validation gap. S1's maximum validation accuracy occurs first at epoch 3. Checkpoint selection uses validation accuracy, earliest on ties. We selected S1 before evaluating held-out test predictions. Perfect validation accuracy does not establish generalization, especially on this correlated dataset.\nSources: runs/S1_limited_v1/frozen_history.json; runs/A0_limited_v1/frozen_history.json; artifacts/evaluation/selection.json. Values are accuracy fractions from 0 to 1.")
 curve(s,'S1',64,154,552,406);curve(s,'A0',664,154,552,406)
 box(s,64,592,1152,52,'We selected the checkpoint and app model using validation before test evaluation.',25,MUTED)
 
 s=new('MobileNetV2 scored highest on the test set',"All four models use exactly the same 51 held-out test images. These are cleaned-data retrainings of the original CNN and MobileNetV2 approaches, not historical A1 scores on contaminated full-frame images. S1 achieves 78.43% accuracy and macro-F1 0.7300. B1 performs better on test, but switching the deployed model after seeing test results would invalidate the recorded selection protocol. A0/S1 isolates pretraining, not architectural superiority. The accuracy improvement is 37.25 percentage points over A0. Single-seed results and small correlated test support limit conclusions.\nSource: artifacts/evaluation/comparison.json; artifacts/evaluation/selection.json. Exact values appear in the JSON; slides round accuracy to 2 decimals and F1 to 4.")
 names={'B0':'Simple CNN','B1':'MobileNetV2','S1':'ConvNeXt pretrained','A0':'ConvNeXt random'}
 values=[['Model','Accuracy','Macro-F1']]
 for r in ['B0','B1','S1','A0']:
  m=results[r]['overall'];values.append([r+'  '+names[r],f'{m["accuracy"]*100:.2f}%',f'{m["macro_f1"]:.4f}'])
 table(s,values,64,170,[672,240,240],68)
 delta=100*(results['S1']['overall']['accuracy']-results['A0']['overall']['accuracy'])
 box(s,64,548,1152,50,f'Pretraining improved ConvNeXt accuracy by {delta:.2f} percentage points.',29,ACCENT)
 box(s,64,606,1152,40,'S1 remains the app model selected on validation.',25,MUTED)
 
 s=new('Grad-CAM also exposes a real failure',"This is held-out peace_0018. The image is labeled peace, but the selected model predicts stop. Stop probability is about 44.3%, and peace is about 44.0%, so the decision is close. Grad-CAM targets the predicted class logit and the spatial backbone features. We checked that maps are finite, nonzero and change when the target or input changes. The maps sometimes include the face/background. A heatmap does not establish causal reasoning or correct hand detection. This is a closed-set classifier and can confidently label unrelated images.\nSources: runs/S1_limited_v1/predictions.json; artifacts/evaluation/explanations.json. Images: data/clean/a1_clean_v1/peace/peace_0018.png and artifacts/evaluation/peace_0018_gradcam.png.\nGrad-CAM primary source: https://arxiv.org/abs/1610.02391.")
 image(s,'data/clean/a1_clean_v1/peace/peace_0018.png',64,170,552,332,'Cleaned peace input')
 image(s,'artifacts/evaluation/peace_0018_gradcam.png',664,170,552,332,'Grad-CAM for incorrect stop prediction')
 box(s,64,524,552,44,'True label: peace',28)
 box(s,664,524,552,44,'Prediction: stop',28,ACCENT)
 box(s,64,592,1152,54,'The heatmap includes the hand and face. Background correlations remain possible.',25,MUTED)
 
 s=new('Latency, consent and evaluation limits',"The recorded evaluation uses batch size one, ten warmups and one hundred materialized TensorFlow CPU calls. Classifier mean is 395.36 ms, median 393.49 ms and p95 421.26 ms. Grad-CAM takes about 1.9 seconds and is measured separately. These are not camera FPS or full request latency. UI timings can vary between calls.\nThe user confirmed project-use consent for the teammate and all visible background classmates. Public redistribution is separate and remains unauthorized. The app processes uploads in memory. We used zero synthetic training images. Stronger claims need independent participants, sessions and environments; expanding the dataset was deferred. Avoid drawing conclusions from only four stop test images.\nSources: artifacts/evaluation/latency.json; artifacts/audit/consent_scope.json; docs/dataset_card.md.")
 box(s,64,176,565,112,'395.36 ms',80,ACCENT,font='Georgia')
 box(s,64,320,550,108,'Mean CPU prediction time\n100 calls after 10 warmups',29)
 box(s,64,465,550,96,'Grad-CAM takes about 1.9 seconds\nand is measured separately.',25,MUTED)
 box(s,710,176,506,92,'One participant, one indoor setting.\nOnly four stop test examples.',27)
 box(s,710,328,506,92,'Course-use consent confirmed.\nPublic redistribution is separate.',27)
 box(s,710,480,506,80,'Zero synthetic training images.\nUploads remain in memory.',27)

 s=new('Live demonstration',"Switch to the browser at http://127.0.0.1:8765. If the server is not running, use the project directory and command shown below. Existing .venv and selected weights are already available. Upload peace_0000, stop_0024 and thumbs_up_0064; use peace_0018 to demonstrate a failure. Do not promise a correct result for an unseen hand. The report has seven pages. The demo's silent 4:10 master needs the students' own audio, using the timestamped Markdown transcript. No publication, submission or defense completion is implied by local artifacts.\nStart: cd /home/nurik/Projects/cv-assignment\n.venv/bin/python demo.py --bundle deliverables/model_bundle.json\nFiles: output/pdf/A2_Gesture_Recognition_Report.pdf; deliverables/A2_Demo_Silent.mp4; deliverables/A2_Demo_Transcript.md; deliverables/A2_Source.zip.")
 box(s,64,170,1152,80,'http://127.0.0.1:8765',48,ACCENT)
 box(s,64,296,1152,64,'Peace, stop and thumbs-up examples are ready in the dataset folder.',28)
 box(s,64,396,1152,72,'A real failure is ready too: peace_0018.jpg predicts stop.',28)
 box(s,64,528,1152,100,'The report records the full protocol, training curves and per-class results.\nSpeaker notes include sources and the local startup command.',25,MUTED)
 
 doc.DocumentProperties.Title='Gesture Lab - Assignment 2 defense'
 doc.DocumentProperties.Subject='Hand gesture recognition with ConvNeXt and Grad-CAM'
 doc.DocumentProperties.Author='Gesture Lab'
 BUILD.mkdir(parents=True,exist_ok=True)
 # Store editable source and PPTX, then reopen the actual PPTX before rendering.
 doc.storeAsURL(url(BUILD/'defense_source.odp'),(prop('FilterName','impress8'),prop('Overwrite',True)))
 doc.storeToURL(url(OUT),(prop('FilterName','Impress MS PowerPoint 2007 XML'),prop('Overwrite',True)))
 deck=desktop.loadComponentFromURL(url(OUT),'_blank',0,(prop('Hidden',True),))
 doc.close(True)
 assert deck.getDrawPages().Count==9
 deck.storeToURL(url(BUILD/'defense_render.pdf'),(prop('FilterName','impress_pdf_Export'),prop('Overwrite',True)))
 deck.close(True)
 (BUILD/'slide_sources.json').write_text(json.dumps(evidence,indent=2)+'\n')
 print('Created and reopened 9-slide PPTX:',OUT,flush=True)
if __name__=='__main__':main()
