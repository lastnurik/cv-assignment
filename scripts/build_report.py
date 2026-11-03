"""Seven-page assignment report derived from frozen experiment artifacts."""
import json
from pathlib import Path
from xml.sax.saxutils import escape
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet,ParagraphStyle
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Table,TableStyle,Image,PageBreak
from reportlab.graphics.shapes import Drawing,Rect,String,Line,Polygon
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from PIL import Image as PILImage

ROOT=Path('.');OUT=Path('output/pdf');E=Path('artifacts/evaluation')
def read(path):return json.loads(Path(path).read_text())
def main():
    OUT.mkdir(parents=True,exist_ok=True)
    results=read(E/'comparison.json');selection=read(E/'selection.json');latency=read(E/'latency.json');explanations=read(E/'explanations.json');qa=read('artifacts/qa/browser_qa.json')
    assert qa['passed'] and read(E/'verification.json')['passed']
    pdfmetrics.registerFont(TTFont('Lato','/usr/share/fonts/truetype/lato/Lato-Regular.ttf'))
    pdfmetrics.registerFont(TTFont('LatoBold','/usr/share/fonts/truetype/lato/Lato-Bold.ttf'))
    pdfmetrics.registerFont(TTFont('LatoItalic','/usr/share/fonts/truetype/lato/Lato-Italic.ttf'))
    pdfmetrics.registerFontFamily('Lato',normal='Lato',bold='LatoBold',italic='LatoItalic',boldItalic='LatoBold')
    pdfmetrics.registerFont(TTFont('Editorial','/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf'))
    styles=getSampleStyleSheet();styles.add(ParagraphStyle(name='BodyA2',fontName='Lato',fontSize=10,leading=14,spaceAfter=9,textColor=colors.HexColor('#253142')))
    styles['Heading1'].fontName='Editorial';styles['Heading1'].textColor=colors.HexColor('#172d37');styles['Heading1'].fontSize=21;styles['Heading1'].leading=25
    styles['Heading2'].fontName='LatoBold';styles['Heading2'].textColor=colors.HexColor('#a34329');styles['Heading2'].fontSize=13
    story=[]
    def p(text):story.append(Paragraph(text,styles['BodyA2']))
    def h(text):story.append(Paragraph(text,styles['Heading1']));story.append(Spacer(1,10))
    def sub(text):story.append(Paragraph(text,styles['Heading2']))
    def table(rows,widths=None):
        data=[[Paragraph(escape(str(x)),styles['BodyA2']) for x in row] for row in rows]
        t=Table(data,colWidths=widths,repeatRows=1,hAlign='LEFT');t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#eef0eb')),('VALIGN',(0,0),(-1,-1),'TOP'),('LINEBELOW',(0,0),(-1,0),.7,colors.HexColor('#a34329')),('LINEBELOW',(0,1),(-1,-1),.3,colors.HexColor('#d5dce0')),('LEFTPADDING',(0,0),(-1,-1),7),('RIGHTPADDING',(0,0),(-1,-1),7),('TOPPADDING',(0,0),(-1,-1),6),('BOTTOMPADDING',(0,0),(-1,-1),4)]));story.append(t);story.append(Spacer(1,10))
    def fig(path,w,hgt):
        iw,ih=PILImage.open(path).size
        scale=min(w/iw,hgt/ih)
        story.append(Image(str(path),width=iw*scale,height=ih*scale));story.append(Spacer(1,8))
    def page():story.append(PageBreak())
    h('Gesture Lab: hand gesture recognition')
    p('Assignment 2 | Astana IT University | Computer Vision<br/>Instructor: Nurym Kaiyrkhan | Experiments: 4 November 2026')
    sub('Objective and delivered interaction')
    p('This project continues the Assignment 1 classification task: peace, stop and thumbs_up. The web app accepts an uploaded image or a camera capture, returns the three class probabilities, and overlays a class-specific Grad-CAM explanation. The experiment adds ImageNet-pretrained ConvNeXtTiny and compares it with clean re-trained versions of the A1 simple CNN and MobileNetV2 approaches. A matched ablation replaces the pretrained ConvNeXt backbone with a randomly initialized frozen backbone.')
    fig('artifacts/qa/browser/app_workspace.png',490,318)
    p('Figure 1. Actual local app after an uploaded peace image. The app reports the active model, inference and explanation time, and its closed-set limitation. The screenshot shows an actual model response.')
    p('The selected ConvNeXt model achieved 78.43% test accuracy and macro-F1 0.7300. MobileNetV2 achieved higher test accuracy (86.27%). ConvNeXt was selected using validation only; the test result did not trigger model switching or additional tuning. These results describe a small, correlated, single-participant indoor dataset, not generalization to new people.')
    page();h('Dataset, cleaning and evaluation protocol')
    p('A1 contains 464 JPEGs at 640 x 480: 151 peace, 163 stop and 150 thumbs_up. The original collector burned class labels into saved images. All original files and models were preserved with SHA-256 inventory checks. A fixed top-20% crop removes 96 rows from every image, independent of class. The resulting 640 x 384 RGB PNG is resized to 224 x 224. Normalization belongs to each model; no second softmax is applied at inference.')
    p('All original contact sheets were inspected. Eleven stop images (stop_0152 through stop_0162) without a visible gesture were excluded without relabeling. The retained set contains 453 images. Capture storage now saves clean frames before display overlays and uses unique session directories, preventing overwrites.')
    table([['Class','Train','Validation','Test','Total'],['peace',105,23,23,151],['stop',136,12,4,152],['thumbs_up',103,23,24,150],['Total',344,58,51,453]],[150,80,90,80,90])
    sub('What the split does and does not establish')
    p('We continued with the existing dataset and deferred new collection. Original session IDs remain unknown. Connected components of the pre-existing 64-bit difference-hash graph (Hamming distance at most 4) form conservative similarity groups. All 38 components, including mixed-class components, remain intact within a split. Known similarity edges, exact duplicate hashes and source/derivative hashes have zero cross-split overlap. No fabricated session IDs were introduced.')
    p('The largest component goes to training; remaining whole components are allocated deterministically toward train/validation/test proportions. This prevents known duplicate overlap but cannot remove common-subject, room or capture-session correlations. A random independent sample assumption is not justified. Only four stop examples are in the test set, so that class estimate is particularly uncertain. No person-independent, session-independent or varied-environment robustness claim is made.')
    p('Frozen dataset: data/manifests/a1_limited_v2.json. The same train, validation and test IDs are used for every model. No synthetic samples are used. Test inference occurs only after validation checkpoint and deployment selection are recorded. Three pre-split images were used for disposable feasibility checks; their pilot heads were discarded and final experiments initialized independently.')
    page();h('Method: ConvNeXt in one page')
    sub('Modernizing a convolutional network')
    p('ConvNeXt was introduced in the CVPR 2022 paper A ConvNet for the 2020s [1]. Rather than adding self-attention, the authors systematically modernized a conventional convolutional architecture using design choices associated with contemporary vision models: a patch-like stem, changed stage ratios, depthwise large-kernel convolutions, an inverted bottleneck, LayerNorm and GELU. The paper establishes its state-of-the-art lineage at publication; this report does not claim ConvNeXtTiny is the best model available in 2026.')
    p('Neurohive\'s CNN guide [2] explicitly discusses ConvNeXt and its comparison with Transformers. The primary paper and official implementation [1,3] control the technical description. Tiny is a smaller member of the family, suitable for a local CPU demonstration. The Keras 3.15.1 backbone has 27,820,128 parameters. We downloaded its ImageNet weights from the official Keras endpoint and verified their checksum [4].')
    d=Drawing(490,110)
    boxes=[(0,'RGB image','crop + resize'),(126,'ConvNeXtTiny','frozen backbone'),(252,'Average pooling','768 features'),(378,'Dense + softmax','3 gesture scores')]
    for x,a,b in boxes:
        d.add(Rect(x,35,112,60,rx=6,fillColor=colors.HexColor('#eef0eb'),strokeColor=colors.HexColor('#a34329')));d.add(String(x+56,72,a,textAnchor='middle',fontSize=9));d.add(String(x+56,54,b,textAnchor='middle',fontSize=8))
        if x<378:d.add(Line(x+113,65,x+124,65,strokeColor=colors.HexColor('#172d37')))
    story.append(d)
    p('Figure 2. The selected classification pipeline. A spatial 7 x 7 x 768 backbone output is retained for Grad-CAM; global average pooling supplies the classifier features. Keras ConvNeXt includes input normalization, so it receives RGB float32 pixels in [0,255] without MobileNet-style rescaling.')
    sub('Hypothesis and controlled ablation')
    p('The hypothesis is that general visual features learned from ImageNet help classify gestures with limited local data. S1 freezes an ImageNet-initialized backbone and trains a three-class head. A0 freezes the same architecture with random backbone weights and trains an identically initialized head. Both use seed 123, Dense initializer seed 223, Adam at 0.001, batch size 16, 22 steps per epoch and 20 epochs, with no augmentation. This isolates the contribution of pretrained representation weights. It does not test architecture superiority or full end-to-end fine-tuning.')
    p('Frozen train/validation features are cached once per run to avoid repeated CPU backbone execution. Test features are not extracted during training. The full spatial model is saved at each validation improvement, preserving equivalent inference and explanation behavior. Cached-head versus reloaded full-model probabilities were checked numerically. No synthetic training images are used.')
    page();h('Training controls and learning curves')
    table([['Run','Architecture / initialization','Budget'],['B0','A1 simple CNN; random','10 epochs'],['B1','A1 MobileNetV2; ImageNet','5 frozen + 3 fine-tune'],['S1','ConvNeXtTiny; ImageNet; frozen','20 head epochs'],['A0','ConvNeXtTiny; random; frozen','20 head epochs']],[55,285,150])
    p('All models use seed 123, batch size 16 and 22 steps per epoch. B0/B1 retain the A1-style flip, rotation, zoom and translation augmentation on training only. MobileNetV2 fine-tunes its last 30 layers except BatchNormalization at learning rate 0.00001; frozen training uses 0.001. B0/B1 are architecture comparisons, not a single-factor ablation. Only A0/S1 have matched architecture, head initialization, augmentation policy and training budget.')
    fig(E/'training_curves.png',460,280)
    p('Figure 3. Saved train/validation accuracy histories, including distinct MobileNet frozen and fine-tuning stages. Loss curves are also supplied as artifacts/evaluation/loss_curves.png. The checkpoint is the maximum validation-accuracy epoch, earliest on ties; best and last checkpoints are stored separately. ConvNeXt S1 reached validation accuracy 1.0 at epoch 3; this does not imply perfect test performance.')
    durations=[['Run','Training incl. extraction (s)','Selected phase / epoch']]
    for role in ['B0','B1','S1','A0']:
        summary=read(f'runs/{role}_limited_v1/summary.json');r=selection['runs'][role];durations.append([role,f"{summary['training_seconds']:.1f}",f"{r['selected_phase']} / {r['selected_epoch']}"])
    p('Training times including feature extraction: '+ '; '.join(f'{row[0]} {row[1]} s (selected {row[2]})' for row in durations[1:])+'.')
    page();h('Results, ablation and limitations')
    rows=[['Run','Accuracy','Macro-F1','Test support']]
    for role in ['B0','B1','S1','A0']:
        m=results[role]['overall'];rows.append([role,f"{100*m['accuracy']:.2f}%",f"{m['macro_f1']:.4f}",m['samples']])
    table(rows,[70,140,140,140]);fig(E/'confusion_matrices.png',490,343)
    p('Figure 4. Raw confusion matrices on the identical 51 real test images (row=true, column=prediction). Accuracy and macro-F1 were independently recomputed with scikit-learn; differences were below 1e-6. Precision, recall, F1 and support for each class are preserved in comparison.json; zero-division policy is 0.')
    delta=100*(results['S1']['overall']['accuracy']-results['A0']['overall']['accuracy'])
    p(f'The pretraining ablation improves accuracy by {delta:.2f} percentage points over frozen random ConvNeXt features. S1 also outperforms the clean simple CNN, but underperforms MobileNetV2 on this test. This is a mixed outcome, not a universal upgrade claim. S1 remains the app model because its validation score selected it before test access.')
    p('The environment report contains one group, original_indoor_room, with all 51 test images. There is no measured cross-environment robustness result. Single-seed training, correlation between captures, uneven class support and unvalidated similarity-group independence limit statistical conclusions. Historical A1 accuracies from contaminated full-frame images are not treated as directly comparable benchmarks.')
    page();h('Explainability and interactive performance')
    cells=[]
    for e in explanations:
        title=(e['label'].replace('_',' ').capitalize()+' predicted as '+e['prediction'].replace('_',' '))
        cells.append([Image(e['file'],width=225,height=135),Paragraph(escape(title),styles['BodyA2'])])
    t=Table([cells[:2],cells[2:4]],colWidths=[245,245])
    t.setStyle(TableStyle([('VALIGN',(0,0),(-1,-1),'TOP'),('LEFTPADDING',(0,0),(-1,-1),0),('BOTTOMPADDING',(0,0),(-1,-1),8)]))
    story.append(t)
    p('Figure 5. Actual Grad-CAM overlays for the three classes and a failed peace prediction. Gradients target the pre-softmax class logit and spatial backbone output. Maps were finite, nonzero and changed with the target class. Raw-source and cleaned-image probabilities matched within atol=1e-5, rtol=1e-4. Overlays align with the cropped input.')
    failure=next((x for x in explanations if x['label']!=x['prediction']),None)
    if failure:p(f"An observed error, {failure['sample_id']}, was labeled {failure['label']} but predicted {failure['prediction']}. Its saved heatmap is included with the evidence. Attention sometimes includes the face/background as well as the hand; Grad-CAM is diagnostic, not proof of causal reasoning.")
    table([['CPU classifier timing','Milliseconds'],['Mean (100 calls)',f"{latency['mean_ms']:.2f}"],['Median',f"{latency['median_ms']:.2f}"],['95th percentile',f"{latency['p95_ms']:.2f}"]],[300,190])
    p('Timing uses batch size 1, ten warmups and 100 materialized TensorFlow CPU calls. TensorFlow 2.21.0, Python 3.12.15, Linux x86_64, four intra-op/two inter-op threads. Explanation takes approximately 1.9 seconds in the recorded evaluation and is measured separately. These figures are not camera FPS or complete request latency. Actual browser request timings are recorded in browser_qa.json.')
    p('Chrome upload tests exercise all three labels, a genuine classification error, invalid-image handling and clear/reset. Headless camera denial/unavailability displays an upload fallback. Physical-camera capture is unverified; the working upload route satisfies the required upload/camera interaction. An unrelated image can still receive a confident class because no no-hand detector is implemented.')
    page();h('Ethics, reproducibility and delivery')
    p('Project-use consent was confirmed for the participant and all visible background classmates. Public redistribution of identifiable images is not assumed. Source images, contact sheets and trained artifacts should be shared only with the intended course reviewer under that permission. The local app processes uploads/camera frames in memory and does not save them or log image payloads.')
    p('We used no synthetic training images. The model is limited to three gestures and must not be used for consequential decisions about people. Bias toward the observed participant, background, camera and gesture style is expected. Future evaluation needs independent participants, sessions and environments. That collection was deferred.')
    sub('Reproduce the release')
    p('Use Python 3.12 and requirements-runtime.lock.txt. Install dependencies, then start: <font face="Courier">python demo.py --bundle deliverables/model_bundle.json</font>. Open the printed localhost address. The bundle contains a relative checkpoint path, class order, preprocessing and SHA-256. Final source/package instructions are in README.md and deliverables/RELEASE_GUIDE.md. The demo is supplied as a silent recording with a timestamped voiceover script for the students. Repository and video links will be shared with the course reviewer after the students record their audio.')
    p('Frozen run configurations are configs/{B0,B1,S1,A0}.convnext_v1.json. Each run stores initialization, best/last checkpoints, phase histories, configuration and data hash. Per-image predictions and labels are retained for metric recomputation. Run scripts/evaluate_predictions.py on the four predictions files to reproduce comparisons; scripts/plot_release.py rebuilds figures. Do not overwrite run IDs or silently retrain after inspecting test results.')
    sub('References')
    refs=[('[1] Liu et al. A ConvNet for the 2020s. CVPR, 2022.','https://arxiv.org/abs/2201.03545'),('[2] Neurohive. Convolutional Neural Network guide (ConvNeXt discussion).','https://neurohive.io/en/guides/convolutional-neural-network/'),('[3] Official ConvNeXt implementation and MIT license.','https://github.com/facebookresearch/ConvNeXt'),('[4] Keras ConvNeXt API and preprocessing documentation.','https://keras.io/api/applications/convnext/'),('[5] Selvaraju et al. Grad-CAM: Visual Explanations from Deep Networks. ICCV, 2017.','https://arxiv.org/abs/1610.02391')]
    for name,url in refs:p(f'{escape(name)} <link href="{url}" color="#a34329">{escape(url)}</link>')
    def footer(canvas,doc):
        canvas.setStrokeColor(colors.HexColor('#d5d9d6'));canvas.line(42,39,A4[0]-42,39);canvas.setFont('Lato',8);canvas.setFillColor(colors.HexColor('#657281'));canvas.drawString(42,25,'GESTURE LAB  /  A2                                      One participant, one setting');canvas.drawRightString(A4[0]-42,25,str(doc.page))
    doc=SimpleDocTemplate(str(OUT/'A2_Gesture_Recognition_Report.pdf'),pagesize=A4,leftMargin=42,rightMargin=42,topMargin=36,bottomMargin=42,title='Gesture Lab - Assignment 2',author='Gesture Lab project')
    doc.build(story,onFirstPage=footer,onLaterPages=footer)
    from pypdf import PdfReader
    n=len(PdfReader(OUT/'A2_Gesture_Recognition_Report.pdf').pages);assert 6<=n<=8,n
    print('Report pages:',n)
if __name__=='__main__':main()
