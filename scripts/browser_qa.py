"""Actual Chrome upload/error/fallback checks against the selected local app."""
import json,time
from pathlib import Path
from playwright.sync_api import sync_playwright
from gesture_cv.dataset import write_json,sha256

def main():
    out=Path('artifacts/qa/browser');out.mkdir(parents=True,exist_ok=True)
    records=[]
    with sync_playwright() as p:
        browser=p.chromium.launch(executable_path='/usr/bin/google-chrome',headless=True)
        context=browser.new_context(viewport={'width':1440,'height':1000})
        page=context.new_page()
        page.add_init_script("""const originalFetch=window.fetch;window.fetch=async(...args)=>{const response=await originalFetch(...args);if(String(args[0]).includes('/api/predict')){window.lastPrediction=await response.clone().json();}return response;};""")
        errors=[];page.on('pageerror',lambda error:errors.append(str(error)))
        page.goto('http://127.0.0.1:8765');page.locator('#file').wait_for()
        for label,name in [('peace','peace_0000'),('stop','stop_0024'),('thumbs_up','thumbs_up_0064'),('peace','peace_0018')]:
            page.locator('#file').set_input_files(f'dataset/{label}/{name}.jpg')
            start=time.perf_counter()
            with page.expect_response('**/api/predict',timeout=90000) as response:
                page.locator('#predict').click()
            r=response.value;assert r.status==200
            page.wait_for_function('window.lastPrediction && window.lastPrediction.prediction')
            data=page.evaluate('window.lastPrediction')
            page.locator('#result').wait_for(state='visible');assert page.locator('#label').inner_text()==data['prediction'].replace('_',' ')
            assert page.locator('#heatmap').get_attribute('src')==data['heatmap']
            page.screenshot(path=str(out/(name+'.png')),full_page=True)
            records.append({'sample_id':name,'prediction':data['prediction'],'probabilities':data['probabilities'],'response_status':r.status,'end_to_end_ms':(time.perf_counter()-start)*1000,'model':data['model']})
        # A constructed unrelated image must yield a closed-set output, not be claimed hand detection.
        page.locator('#file').set_input_files({'name':'unrelated.svg','mimeType':'image/svg+xml','buffer':b'<svg></svg>'})
        with page.expect_response('**/api/predict',timeout=90000) as response:page.locator('#predict').click()
        assert response.value.status==400;page.locator('#error').wait_for(state='visible')
        invalid=page.locator('#error').inner_text();assert invalid
        page.locator('#clear').click();assert page.locator('#predict').is_disabled()
        page.locator('#camera').click();page.wait_for_function("document.getElementById('error').textContent.includes('Camera unavailable')",timeout=15000)
        camera_error=page.locator('#error').inner_text();assert 'upload' in camera_error.lower()
        page.screenshot(path=str(out/'camera_fallback.png'),full_page=True)
        assert not errors,errors
        write_json('artifacts/qa/browser_qa.json',{'passed':True,'browser_version':browser.version,'user_agent':page.evaluate('navigator.userAgent'),'bundle_sha256':sha256('deliverables/model_bundle.json'),'records':records,'invalid_upload_error':invalid,'camera_fallback':camera_error,'actual_camera_capture':'not tested; upload route provides required interaction','clear_passed':True,'javascript_errors':errors})
        context.close();browser.close()
if __name__=='__main__':main()
