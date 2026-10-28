"""Validate published local navigation/assets and reject excluded files."""
import argparse
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urlsplit,unquote
class Parser(HTMLParser):
 def __init__(self):super().__init__();self.links=[];self.ids=set()
 def handle_starttag(self,tag,attrs):
  for key,value in attrs:
   if key in ('href','src') and value:self.links.append(value)
   if key=='id':self.ids.add(value)
def validate(root,prefix):
 errors=[];count=0
 for p in root.rglob('*'):
  if p.is_file() and (p.name in ('PLAN.md','ML4HL-Widgets.html','.env.local','mcp.json') or 'ML4HL-Motif' in p.name):errors.append(f'Excluded file: {p}')
 for p in root.rglob('*.html'):
  parser=Parser();parser.feed(p.read_text());count+=1
  for link in parser.links:
   u=urlsplit(link)
   if u.scheme or u.netloc or not u.path:continue
   path=unquote(u.path)
   if path.startswith('/'):
    if not path.startswith(prefix):errors.append(f'{p}: outside project prefix: {link}');continue
    target=root/path[len(prefix):]
   else:target=p.parent/path
   if target.is_dir():target=target/'index.html'
   if not target.is_file():errors.append(f'{p.relative_to(root)}: missing {link}')
 if errors:raise SystemExit('\n'.join(errors))
 print(f'PASS: {count} HTML pages, local assets/navigation under {prefix}, publication exclusions')
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('site',type=Path);ap.add_argument('--prefix',required=True);a=ap.parse_args();validate(a.site.absolute(),a.prefix)
