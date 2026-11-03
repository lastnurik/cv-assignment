"""Record real app interactions; edit only scene length. No synthetic narration."""
import json, subprocess, time, shutil
from pathlib import Path
from playwright.sync_api import sync_playwright
from gesture_cv.dataset import sha256, write_json

ROOT=Path('.').resolve()
OUT=ROOT/'artifacts/qa/refinement_v2/video'
OUT.mkdir(parents=True,exist_ok=True)
PRESENT=ROOT/'deliverables/presentation'
PRESENT.mkdir(parents=True,exist_ok=True)
SCENES=[
('intro',24,'The project',"""This is Gesture Lab, our hand gesture recognition project. We kept the Assignment One topic and dataset: peace, stop, and thumbs up. For Assignment Two, we added ConvNeXt, compared it with our original model approaches, and built this local app with visual explanations."""),
('peace',36,'Upload and explain',"""I’ll start with a peace image. The original photo has a collection label at the top. Before prediction, the pipeline removes the same top twenty percent from every image, so that label does not reach the model. We then get probabilities for all three classes and a Grad-CAM heatmap. The warmer areas indicate regions that contribute to this class score. Here, peace has the highest probability, but the model is not very certain."""),
('stop',26,'A stop gesture',"""Here is a stop gesture. It goes through the same crop and preprocessing, and the app predicts stop. The timing panel separates the prediction from the explanation, because calculating the heatmap takes longer. These are measured request times. Uploaded photos stay in memory on the local server and are not saved."""),
('thumb',26,'A thumbs-up gesture',"""Next is thumbs up. The predicted label is correct, but the heatmap also includes the face and background. That matters: a correct answer does not prove the model has learned only the hand gesture. The app supports camera capture too. We verified upload and the fallback when camera access is unavailable."""),
('failure',34,'An actual error',"""This example shows a real failure. The image is labeled peace, but the model predicts stop. The two probabilities are almost tied, so the result is uncertain even though the interface must choose one label. We kept this failure in the demonstration. Our data contains one main participant in one room. Similarity groups stay together across the split, but we cannot claim the model will generalize to new people or environments."""),
('method',34,'Method and controlled ablation',"""Our modern component is ConvNeXtTiny, from the ConvNeXt architecture introduced at CVPR in twenty twenty-two. We freeze the backbone and train a three-class head. To test the value of pretraining, we compare ImageNet weights with a randomly initialized frozen ConvNeXt backbone. The architecture, head initialization, optimizer, seed, and twenty-epoch head-training budget are the same. Only backbone initialization changes. The original CNN and MobileNet use the same frozen data split."""),
('results',40,'Measured results',"""All four models were evaluated on the same fifty-one test images. The simple CNN scored forty-seven point zero six percent accuracy. MobileNet reached eighty-six point two seven percent. Pretrained ConvNeXt reached seventy-eight point four three percent, with macro F one of zero point seven three. Random frozen ConvNeXt reached forty-one point one eight percent. Pretraining clearly helped in this experiment, but ConvNeXt did not outperform MobileNet. We kept ConvNeXt in the app because it was selected using validation before we evaluated the test set."""),
('limits',30,'Limits and deliverables',"""The measured mean prediction time was about three hundred ninety-five milliseconds on our CPU, excluding the heatmap. The main limitations are the single participant, shared setting, and only four stop images in the test set. Everyone pictured consented to course project use. We used no synthetic training images. Our deliverables are the working app, seven-page report, repository package, and this demonstration. The report includes training curves, class results, the ablation, and these limitations.""")
]
def stamp(t):return f'{int(t)//60:02}:{int(t)%60:02}'
def create_materials():
    results=json.loads((ROOT/'artifacts/evaluation/comparison.json').read_text())
    base="""<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Gesture Lab | Presentation</title><style>
    *{box-sizing:border-box}body{margin:0;background:#f5f4ef;color:#172d37;font-family:Lato,'Segoe UI',sans-serif}main{padding:60px 76px;max-width:1600px;margin:auto;min-height:100vh}header{display:flex;justify-content:space-between;border-bottom:1px solid #cbd3ce;padding-bottom:24px;font-size:17px;color:#59676b}header b{font:32px Georgia;color:#172d37}.kicker{margin-top:42px;font-size:15px;letter-spacing:.1em;text-transform:uppercase;color:#a34329}h1{font:normal 58px/1.12 Georgia;letter-spacing:-1.7px;margin:16px 0 24px;max-width:1100px}p{font-size:23px;line-height:1.6;max-width:1170px}.split{display:grid;grid-template-columns:1.2fr 1fr;gap:60px;margin-top:34px}.step{display:flex;align-items:baseline;gap:25px;padding:20px 0;border-top:1px solid #cbd3ce}.step span{font-size:16px;color:#a34329}.step strong{font-size:26px;font-weight:500}.step small{margin-left:auto;font-size:17px;color:#59676b}.aside{border-left:2px solid #a34329;padding-left:28px}.aside h2{font:32px Georgia;margin:0 0 20px}.aside p{font-size:21px}table{width:100%;border-collapse:collapse;margin-top:26px;font-variant-numeric:tabular-nums}th{text-align:left;color:#59676b;font-size:16px;font-weight:500}td{font-size:24px}th,td{padding:19px 18px;border-bottom:1px solid #cbd3ce}td:first-child,th:first-child{padding-left:0}tr.selected{background:#e7ece9}.num{text-align:right}.note{color:#59676b;font-size:18px}.bottom{border-top:1px solid #cbd3ce;margin-top:34px;padding-top:20px;font-size:17px;color:#59676b}.measure{font:76px Georgia;color:#a34329;margin:24px 0 6px}.support{font-size:20px;color:#59676b}.links{display:flex;gap:25px;font-size:15px;margin-top:32px}a{color:#59676b} @media(max-width:900px){main{padding:30px}.split{grid-template-columns:1fr}h1{font-size:38px}td{font-size:18px}.step small{margin-left:0}header{font-size:12px}header b{font-size:24px}p{font-size:20px}}
    </style><main><header><b>Gesture Lab</b><span>Astana IT University / Computer Vision / Assignment 02</span></header>"""
    nav='<nav class="links" aria-label="Presentation pages"><a href="method.html">Method</a><a href="results.html">Results</a><a href="limits.html">Limits & delivery</a></nav>'
    pages={}
    pages['method']=base+"""<p class="kicker">Method / ConvNeXtTiny</p><h1>Does pretraining help<br>with a small gesture dataset?</h1><div class="split"><div><div class="step"><span>01</span><strong>RGB image</strong><small>crop + resize</small></div><div class="step"><span>02</span><strong>ConvNeXtTiny</strong><small>frozen backbone</small></div><div class="step"><span>03</span><strong>Average pooling</strong><small>768 features</small></div><div class="step"><span>04</span><strong>Classification head</strong><small>3 gesture classes</small></div></div><div class="aside"><h2>A controlled comparison</h2><p>S1: ImageNet-pretrained backbone<br>A0: randomly initialized backbone</p><p>Same architecture, head initialization, optimizer, seed and 20 training epochs.</p><p class="note">All four models use the same frozen split: 344 train / 58 validation / 51 test.</p></div></div><p class="bottom">ConvNeXt: Liu et al., CVPR 2022. Spatial features retained for Grad-CAM.</p>"""+nav+'</main></html>'
    names={'B0':'Simple CNN','B1':'MobileNetV2','S1':'ConvNeXt / pretrained','A0':'ConvNeXt / random'}
    rows=''.join(f'<tr class="{"selected" if r=="S1" else ""}"><td>{r} &nbsp; {names[r]}</td><td class="num">{100*d["overall"]["accuracy"]:.2f}%</td><td class="num">{d["overall"]["macro_f1"]:.4f}</td></tr>' for r,d in results.items())
    pages['results']=base+f"""<p class="kicker">Results / same 51 test images</p><h1>Pretraining helped.<br>MobileNet performed best on test.</h1><table><thead><tr><th>Model</th><th class="num">Accuracy</th><th class="num">Macro-F1</th></tr></thead><tbody>{rows}</tbody></table><p>S1 improved accuracy by {100*(results['S1']['overall']['accuracy']-results['A0']['overall']['accuracy']):.2f} percentage points over random frozen features.</p><p class="bottom">The app model was selected on validation before test evaluation. No model switch or extra tuning followed the test results.</p>"""+nav+'</main></html>'
    pages['limits']=base+"""<p class="kicker">Limits / ethics / delivery</p><h1>Useful for this setting.<br>New people remain untested.</h1><div class="split"><div><p class="measure">395.36 ms</p><p class="support">Mean CPU prediction time<br>100 calls after 10 warmups; excludes Grad-CAM</p><p class="note">Explanation takes longer and is reported separately in the app.</p></div><div class="aside"><h2>What to keep in mind</h2><p>One participant, one indoor setting.<br>Only four stop images in the test set.</p><p>Consent confirmed for course use.<br>Zero synthetic training images.</p><p>App + 7-page report + source repository<br>+ demonstration with student narration.</p></div></div><p class="bottom">Uploads are processed in memory. Identifiable data stays restricted to permitted course review.</p>"""+nav+'</main></html>'
    for k,v in pages.items():(PRESENT/f'{k}.html').write_text(v)
    timeline=[];start=0
    md=['# Gesture Lab demo transcript','',
        'Video: A2_Demo_Silent.mp4. Total length: 04:10. The video has no audio track. Record the text below in your own voice and place each section at its start timestamp. A short pause at the end of a section is intentional. Do not read the screen directions or headings aloud.','',
        'The recorded clicks and model outputs are real. Each example plays at its recorded speed, then holds on the result to leave time for narration. Presentation screens summarize the saved experiment results.','',
        'Pronunciation: ConvNeXt = “conv next”; Grad-CAM = “grad cam”; macro-F1 = “macro F one”.','',
        '| Time | Screen |','|---|---|']
    for k,d,title,text in SCENES:
        timeline.append({'key':k,'start':start,'end':start+d,'duration':d,'title':title,'text':text})
        md.append(f'| {stamp(start)}–{stamp(start+d)} | {title} |');start+=d
    md+=['','## Voiceover','']
    for x in timeline:
        md.extend([f'### {stamp(x["start"])}–{stamp(x["end"])} | {x["title"]}','',x['text'],''])
    md+=['## Add your recording','',
        'Record in a quiet room at a conversational pace. Keep each paragraph within its time window; the script averages about 125 words per minute. Export one audio file starting at 00:00 and ending at 04:10, then align it with the silent video in your editor. Check the full result once before submission.','',
        'If you use the supplied helper, run it from the project root after recording:','',
        '~~~sh','.venv/bin/python scripts/add_demo_audio.py path/to/your-narration.wav','~~~','',
        'It writes deliverables/A2_Demo_With_Your_Audio.mp4 and leaves the silent master unchanged. The audio must be between 4:00 and 4:10; a shorter final pause is padded with silence. Recording per section and aligning sections in an editor gives the closest match.']
    (ROOT/'deliverables/A2_Demo_Transcript.md').write_text('\n'.join(md)+'\n')
    write_json(OUT/'timeline.json',timeline)
    return timeline

def main():
    timeline=create_materials()
    if __import__('sys').argv[-1]=='--materials-only':return
    clips=[];records=[]
    with sync_playwright() as p:
        b=p.chromium.launch(executable_path='/usr/bin/google-chrome',headless=True)
        for x in timeline:
            key=x['key'];print('Recording',key,flush=True)
            context=b.new_context(viewport={'width':1600,'height':1000},record_video_dir=str(OUT/'raw'),record_video_size={'width':1600,'height':1000})
            page=context.new_page()
            page.add_init_script("""const originalFetch=window.fetch;window.fetch=async(...args)=>{const r=await originalFetch(...args);if(String(args[0]).includes('/api/predict'))window.lastPrediction=await r.clone().json();return r;};""")
            if key in ['method','results','limits']:
                page.goto((PRESENT/f'{key}.html').as_uri())
                page.wait_for_timeout(1800)
                page.screenshot(path=str(OUT/f'{key}.png'))
            else:
                page.goto('http://127.0.0.1:8765')
                page.wait_for_timeout(1500)
                if key!='intro':
                    label,name={'peace':('peace','peace_0000'),'stop':('stop','stop_0024'),'thumb':('thumbs_up','thumbs_up_0064'),'failure':('peace','peace_0018')}[key]
                    page.locator('#file').set_input_files(str(ROOT/f'dataset/{label}/{name}.jpg'))
                    page.wait_for_timeout(1300)
                    with page.expect_response('**/api/predict',timeout=90000) as response:
                        page.locator('#predict').click()
                    assert response.value.status==200
                    page.locator('#result').wait_for(state='visible')
                    data=page.evaluate('window.lastPrediction')
                    records.append({'sample_id':name,'prediction':data['prediction'],'probabilities':data['probabilities'],'inference_ms':data['inference_ms']})
                page.wait_for_timeout(2000)
                page.mouse.move(1500,950)
                page.screenshot(path=str(OUT/f'{key}.png'))
            assert page.evaluate('document.documentElement.scrollWidth<=innerWidth')
            video=page.video;context.close();raw=OUT/f'{key}_raw.webm';video.save_as(str(raw))
            clip=OUT/f'{key}.mp4'
            # Trim startup paint only, retain all interactions at original speed, then hold the result.
            subprocess.run(['ffmpeg','-v','error','-y','-i',str(raw),'-vf',f'trim=start=0.3,setpts=PTS-STARTPTS,fps=25,tpad=stop_mode=clone:stop_duration={x["duration"]}', '-t',str(x['duration']),'-an','-c:v','libx264','-preset','fast','-crf','19','-pix_fmt','yuv420p',str(clip)],check=True)
            clips.append(clip)
        b.close()
    manifest=OUT/'concat.txt';manifest.write_text(''.join("file '"+str(f)+"'\n" for f in clips))
    output=ROOT/'deliverables/A2_Demo_Silent.mp4'
    subprocess.run(['ffmpeg','-v','error','-y','-f','concat','-safe','0','-i',str(manifest),'-c','copy','-movflags','+faststart',str(output)],check=True)
    probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-show_format','-show_streams','-of','json',str(output)]))
    duration=float(probe['format']['duration'])
    assert abs(duration-250)<.2 and len(probe['streams'])==1 and probe['streams'][0]['codec_type']=='video'
    write_json(OUT/'recording.json',{'duration_seconds':duration,'audio_streams':0,'size':[1600,1000],'fps':25,'scenes':timeline,'actual_predictions':records,'sha256':sha256(output),'visual_review':'pending'})
    print('Silent master:',output,duration,flush=True)
if __name__=='__main__':main()
