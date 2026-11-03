"""Embed editable chart-owned XLSX snapshots from the real training histories."""
import io,json,zipfile
from pathlib import Path
import xml.etree.ElementTree as E
ROOT=Path(__file__).resolve().parents[1]
NS={'p':'http://schemas.openxmlformats.org/presentationml/2006/main','a':'http://schemas.openxmlformats.org/drawingml/2006/main','c':'http://schemas.openxmlformats.org/drawingml/2006/chart','r':'http://schemas.openxmlformats.org/officeDocument/2006/relationships','s':'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
for k,v in NS.items():E.register_namespace(k,v)
REL='http://schemas.openxmlformats.org/package/2006/relationships';CT='http://schemas.openxmlformats.org/package/2006/content-types'
def tag(p,n):return '{'+NS[p]+'}'+n
def xml(e):
 ns=e.tag.split('}')[0][1:] if e.tag.startswith('{') else ''
 if ns in [REL,CT]:E.register_namespace('',ns)
 return E.tostring(e,encoding='utf-8',xml_declaration=True)
def workbook(data):
 b=io.BytesIO()
 with zipfile.ZipFile(b,'w',zipfile.ZIP_DEFLATED) as z:
  z.writestr('[Content_Types].xml',f'<Types xmlns="{CT}"><Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/><Default Extension="xml" ContentType="application/xml"/><Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/><Override PartName="/xl/worksheets/sheet1.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/></Types>')
  z.writestr('_rels/.rels',f'<Relationships xmlns="{REL}"><Relationship Id="rId1" Type="{NS["r"]}/officeDocument" Target="xl/workbook.xml"/></Relationships>')
  z.writestr('xl/workbook.xml',f'<workbook xmlns="{NS["s"]}" xmlns:r="{NS["r"]}"><sheets><sheet name="Data" sheetId="1" r:id="rId1"/></sheets></workbook>')
  z.writestr('xl/_rels/workbook.xml.rels',f'<Relationships xmlns="{REL}"><Relationship Id="rId1" Type="{NS["r"]}/worksheet" Target="worksheets/sheet1.xml"/></Relationships>')
  sh=E.Element(tag('s','worksheet'));sd=E.SubElement(sh,tag('s','sheetData'))
  rows=[['Epoch','Training','Validation']]+[[str(i+1),t,v] for i,(t,v) in enumerate(zip(data['accuracy'],data['val_accuracy']))]
  for i,row in enumerate(rows,1):
   rr=E.SubElement(sd,tag('s','row'),r=str(i))
   for j,value in enumerate(row):
    c=E.SubElement(rr,tag('s','c'),r=f'{chr(65+j)}{i}')
    if isinstance(value,str):
     c.set('t','inlineStr');E.SubElement(E.SubElement(c,tag('s','is')),tag('s','t')).text=value
    else:E.SubElement(c,tag('s','v')).text=repr(value)
  z.writestr('xl/worksheets/sheet1.xml',xml(sh))
 return b.getvalue()
def main():
 p=ROOT/'deliverables/A2_Defense_Presentation.pptx'
 with zipfile.ZipFile(p) as z:parts={n:z.read(n) for n in z.namelist()}
 for name,body in list(parts.items()):
  if name.startswith('ppt/slides/slide') and name.endswith('.xml'):
   root=E.fromstring(body)
   for shape in root.findall('.//p:sp',NS):
    xf=shape.find('p:spPr/a:xfrm',NS)
    if xf is None:continue
    off=xf.find('a:off',NS);ext=xf.find('a:ext',NS)
    if off is not None and ext is not None and int(off.get('y'))+int(ext.get('cy'))>6858000:
     assert int(off.get('y'))>6200000
     off.set('y','6286500')
   parts[name]=xml(root)
 chartnames=sorted(n for n in parts if n.startswith('ppt/charts/chart') and n.endswith('.xml'))
 assert len(chartnames)==2
 types=E.fromstring(parts['[Content_Types].xml'])
 for num,(name,role) in enumerate(zip(chartnames,['S1','A0']),1):
  data=json.loads((ROOT/f'runs/{role}_limited_v1/frozen_history.json').read_text())
  chart=E.fromstring(parts[name]);series=chart.findall('.//c:ser',NS);assert len(series)==2
  for j,s in enumerate(series):
   vals=s.find('c:val',NS);cache=vals.find('.//c:numCache',NS)
   if cache is None:cache=vals.find('.//c:numLit',NS)
   assert cache is not None
   actual=[float(pt.find('c:v',NS).text) for pt in cache.findall('c:pt',NS)]
   expected=data['accuracy' if j==0 else 'val_accuracy']
   assert len(actual)==len(expected) and all(abs(a-b)<1e-12 for a,b in zip(actual,expected))
   vals.clear();nr=E.SubElement(vals,tag('c','numRef'));E.SubElement(nr,tag('c','f')).text='Data!$'+chr(66+j)+'$2:$'+chr(66+j)+'$21'
   cache.tag=tag('c','numCache');nr.append(cache)
   cat=s.find('c:cat',NS);cat.clear();ref=E.SubElement(cat,tag('c','strRef'));E.SubElement(ref,tag('c','f')).text='Data!$A$2:$A$21'
   sc=E.SubElement(ref,tag('c','strCache'));E.SubElement(sc,tag('c','ptCount'),val='20')
   for i in range(20):E.SubElement(E.SubElement(sc,tag('c','pt'),idx=str(i)),tag('c','v')).text=str(i+1)
   tx=s.find('c:tx',NS);tx.clear();ref=E.SubElement(tx,tag('c','strRef'));E.SubElement(ref,tag('c','f')).text='Data!$'+chr(66+j)+'$1'
   sc=E.SubElement(ref,tag('c','strCache'));E.SubElement(sc,tag('c','ptCount'),val='1');E.SubElement(E.SubElement(sc,tag('c','pt'),idx='0'),tag('c','v')).text=['Training','Validation'][j]
   sp=s.find('c:spPr',NS)
   if sp is not None:
    ln=sp.find('a:ln',NS)
    if ln is not None:
     for ch in list(ln):
      if ch.tag in [tag('a','solidFill'),tag('a','schemeClr')]:ln.remove(ch)
     fill=E.Element(tag('a','solidFill'));E.SubElement(fill,tag('a','srgbClr'),val=['315C48','A34329'][j]);ln.insert(0,fill)
   marker=s.find('c:marker',NS)
   if marker is not None:
    sym=marker.find('c:symbol',NS)
    if sym is not None:sym.set('val','none')
  old=chart.find('c:externalData',NS)
  if old is not None:chart.remove(old)
  ex=E.Element(tag('c','externalData'),{tag('r','id'):'rIdWorkbook'});E.SubElement(ex,tag('c','autoUpdate'),val='0')
  idx=next((i for i,c in enumerate(list(chart)) if c.tag in [tag('c','printSettings'),tag('c','extLst')]),len(chart));chart.insert(idx,ex)
  parts[name]=xml(chart)
  wbname=f'ppt/embeddings/gesture-chart-{num}.xlsx';parts[wbname]=workbook(data)
  relpath='ppt/charts/_rels/'+Path(name).name+'.rels'
  rels=E.fromstring(parts[relpath]) if relpath in parts else E.Element('{'+REL+'}Relationships')
  E.SubElement(rels,'{'+REL+'}Relationship',Id='rIdWorkbook',Type=NS['r']+'/package',Target=f'../embeddings/gesture-chart-{num}.xlsx')
  parts[relpath]=xml(rels)
  E.SubElement(types,'{'+CT+'}Override',PartName='/'+wbname,ContentType='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
 parts['[Content_Types].xml']=xml(types)
 candidate=ROOT/'artifacts/qa/defense_pptx/final_candidate.pptx'
 with zipfile.ZipFile(candidate,'w',zipfile.ZIP_DEFLATED) as z:
  for n,b in parts.items():z.writestr(n,b)
 candidate.replace(p)
 print('Two editable chart workbooks embedded; values match training histories.')
if __name__=='__main__':main()
