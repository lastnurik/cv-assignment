"""Record actual local app actions with disclosed synthetic narration; no fake predictions."""
import ctypes,ctypes.util,json,time,wave,subprocess
from pathlib import Path
from playwright.sync_api import sync_playwright
from gesture_cv.dataset import write_json

OUT=Path('deliverables/video_work');OUT.mkdir(parents=True,exist_ok=True)
segments=[
('intro','This is Gesture Lab, our Assignment Two hand gesture recognition project. This demonstration uses the actual local web application and actual model outputs. The narration is generated speech. We keep the Assignment One topic: peace, stop, and thumbs up. We cleaned the existing images, compared the original model approaches with Conv Next, and added a controlled pretraining ablation. The application accepts uploaded images, shows class probabilities, and produces a Grad CAM heatmap. It runs locally and does not save uploaded images.'),
('peace','First, we upload a peace gesture. The original image contains the old collection overlay at the top. The inference pipeline removes the same top twenty percent from every image before resizing, so the model does not receive that text. After clicking predict and explain, the server computes real probabilities and a target class heatmap. The displayed confidence is the model probability directly, without applying softmax a second time. The heatmap is diagnostic evidence, not proof that the network understands the gesture.'),
('stop','Next, this stop image passes through the same preprocessing and model. The output includes probabilities for all three supported labels. We display classifier time separately from explanation time because computing gradients is more expensive than a forward prediction. The backend device is reported independently from the browser or camera device. This is a closed set classifier: it always chooses among the supported classes. A confident answer on an unrelated image would not establish that a hand was detected.'),
('thumb','Here is a thumbs up example. The displayed explanation comes from gradients of the predicted class logit with respect to spatial backbone features. The softmax head is not modified to produce the explanation. In verification, the maps were finite, changed with target class and input, and aligned with the cropped image. Upload and simulated camera paths produced equivalent model inputs. The camera interface is available, but the tested browser denied access and correctly offered upload as a fallback.'),
('failure','We also show a genuine failure. This image is labeled peace, but the selected model predicts stop, with close probabilities. We retain this error instead of hiding it or relabeling the image. The heatmap can include the hand, face, and background. The dataset contains one principal participant in one indoor setting, so background correlations remain possible. We kept known similar image groups together when splitting, but similarity grouping does not turn these captures into independent sessions or prove performance on new people.'),
('results','The results shown here come from the saved predictions on the same fifty one test images. The simple convolutional baseline achieved forty seven point zero six percent accuracy. Mobile Net achieved eighty six point two seven percent. Pretrained Conv Next achieved seventy eight point four three percent, with macro F one of zero point seven three. Conv Next with a frozen random backbone achieved forty one point one eight percent. Thus pretraining helped, but Conv Next did not beat Mobile Net on this test. The deployed model was selected using validation before test evaluation.'),
('finish','The ablation keeps the Conv Next architecture, head initialization, seed, optimizer, and head training budget fixed, changing only the backbone initialization. No synthetic training images are used. Average measured CPU classifier latency was about three hundred ninety five milliseconds across one hundred calls after ten warmups; explanation takes longer. Everyone shown consented to course project use. Public redistribution is separate. The report includes curves, confusion matrices, limitations, and sources. The repository package includes reproducible configuration and a local run guide. The small stop test support and lack of diverse environments remain explicit limitations.')]

def synthesize():
    lib=ctypes.CDLL(ctypes.util.find_library('espeak-ng'))
    lib.espeak_Initialize.argtypes=[ctypes.c_int,ctypes.c_int,ctypes.c_char_p,ctypes.c_int];lib.espeak_Initialize.restype=ctypes.c_int
    rate=lib.espeak_Initialize(2,0,None,0);assert rate>0
    chunks=[]
    callback_type=ctypes.CFUNCTYPE(ctypes.c_int,ctypes.POINTER(ctypes.c_short),ctypes.c_int,ctypes.c_void_p)
    @callback_type
    def callback(data,n,events):
        if data and n:chunks.append(ctypes.string_at(data,n*2))
        return 0
    lib.espeak_SetSynthCallback.argtypes=[callback_type];lib.espeak_SetSynthCallback(callback)
    lib.espeak_SetVoiceByName.argtypes=[ctypes.c_char_p];assert lib.espeak_SetVoiceByName(b'en-us')==0
    lib.espeak_SetParameter(1,165,0)
    lib.espeak_Synth.argtypes=[ctypes.c_void_p,ctypes.c_size_t,ctypes.c_uint,ctypes.c_int,ctypes.c_uint,ctypes.c_uint,ctypes.c_void_p,ctypes.c_void_p]
    timeline=[];pcm=b'';offset=0.0
    for key,text in segments:
        chunks.clear();data=text.encode()+b'\0';buf=ctypes.create_string_buffer(data)
        assert lib.espeak_Synth(buf,len(data),0,1,0,1,None,None)==0
        part=b''.join(chunks);duration=len(part)/(rate*2)+2.5
        timeline.append({'key':key,'text':text,'start':offset,'duration':duration});offset+=duration
        pcm+=part+b'\0'*int(2.5*rate)*2
    assert 180<=offset<=300,offset
    with wave.open(str(OUT/'narration.wav'),'wb') as w:w.setnchannels(1);w.setsampwidth(2);w.setframerate(rate);w.writeframes(pcm)
    write_json(OUT/'timeline.json',timeline);return timeline

def main():
    timeline=synthesize()
    comparison=json.loads(Path('artifacts/evaluation/comparison.json').read_text())
    table=''.join(f"<tr><td>{r}</td><td>{100:d['overall']['accuracy']}</td></tr>" for r,d in []) # no hand-entered metrics
    table=''.join(f"<tr><td>{r}</td><td>{100*d['overall']['accuracy']:.2f}%</td><td>{d['overall']['macro_f1']:.4f}</td></tr>" for r,d in comparison.items())
    html=f'''<html><body style="font:24px Arial;padding:55px;background:#f4f7f2;color:#143d4c"><h1>Same 51 test images. Mixed results.</h1><table cellpadding="14"><tr><th>Model</th><th>Accuracy</th><th>Macro-F1</th></tr>{table}</table><p>S1 selected on validation; B1 performs better on test.</p><p>A0/S1: identical frozen architecture and head training; random vs ImageNet backbone.</p><p>One participant / one setting. Stop test support: 4. Zero synthetic training images.</p><p>CPU inference mean: 395.36 ms (100 calls after 10 warmups).</p><p>Project-use consent confirmed. Uploads processed in memory.</p><small>Actual saved experiment results. Generated speech narration disclosed.</small></body></html>'''
    (OUT/'results.html').write_text(html)
    with sync_playwright() as p:
        b=p.chromium.launch(executable_path='/usr/bin/google-chrome',headless=True)
        context=b.new_context(viewport={'width':1440,'height':900},record_video_dir=str(OUT/'recordings'),record_video_size={'width':1440,'height':900})
        page=context.new_page();page.goto('http://127.0.0.1:8765')
        start=time.monotonic()
        for segment in timeline:
            key=segment['key']
            if key in ['peace','stop','thumb','failure']:
                label,name={'peace':('peace','peace_0000'),'stop':('stop','stop_0024'),'thumb':('thumbs_up','thumbs_up_0064'),'failure':('peace','peace_0018')}[key]
                page.locator('#file').set_input_files(f'dataset/{label}/{name}.jpg');page.locator('.workspace').scroll_into_view_if_needed();page.locator('#predict').click();page.locator('#result').wait_for(state='visible',timeout=60000)
            elif key in ['results','finish']:page.goto((OUT/'results.html').resolve().as_uri())
            remaining=segment['start']+segment['duration']-(time.monotonic()-start)
            if remaining>0:page.wait_for_timeout(remaining*1000)
        video=page.video;context.close();video.save_as(str(OUT/'app_recording.webm'));b.close()
    subprocess.run(['ffmpeg','-y','-i',str(OUT/'app_recording.webm'),'-i',str(OUT/'narration.wav'),'-c:v','libx264','-preset','fast','-crf','23','-pix_fmt','yuv420p','-c:a','aac','-shortest','-movflags','+faststart','deliverables/A2_Demo.mp4'],check=True,stdout=subprocess.DEVNULL,stderr=open(OUT/'ffmpeg.log','w'))
    probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-show_format','-show_streams','-of','json','deliverables/A2_Demo.mp4']))
    duration=float(probe['format']['duration']);assert 180<=duration<=300
    assert {'video','audio'}.issubset({x['codec_type'] for x in probe['streams']})
    write_json('artifacts/qa/video_verification.json',{'duration_seconds':duration,'has_audio_video':True,'recording':'Actual local Chrome app actions and actual saved result summary','narration':'Generated eSpeak NG speech, disclosed in narration and final summary','publication':'not performed','visual_audio_review':'pending'})
    print('Demo duration:',duration)
if __name__=='__main__':main()
