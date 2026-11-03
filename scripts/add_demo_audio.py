"""Mux a student's 4:00-4:10 recording with the 4:10 silent master."""
import argparse,json,subprocess
from pathlib import Path

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('audio',type=Path);a=p.parse_args()
    root=Path(__file__).resolve().parents[1];video=root/'deliverables/A2_Demo_Silent.mp4';out=root/'deliverables/A2_Demo_With_Your_Audio.mp4'
    if not a.audio.is_file():p.error('Audio file not found.')
    if out.exists():p.error(f'{out} already exists; rename it before exporting another version.')
    def probe(path):return json.loads(subprocess.check_output(['ffprobe','-v','error','-show_streams','-show_format','-of','json',str(path)]))
    source=probe(a.audio)
    if not any(s['codec_type']=='audio' for s in source['streams']):p.error('The recording has no audio stream.')
    duration=float(source['format']['duration'])
    if not 240<=duration<=250.1:p.error(f'Recording is {duration:.1f}s; align it to the 250s timeline first (accepted length 240-250s).')
    subprocess.run(['ffmpeg','-v','error','-n','-i',str(video),'-i',str(a.audio),'-map','0:v:0','-map','1:a:0','-c:v','copy','-c:a','aac','-b:a','192k','-af','apad','-t','250','-movflags','+faststart',str(out)],check=True)
    final=probe(out);assert {'audio','video'}=={s['codec_type'] for s in final['streams']}
    print(f'Created {out}. Listen and watch once before submission.')
if __name__=='__main__':main()
