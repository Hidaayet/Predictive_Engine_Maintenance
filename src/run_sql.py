import sys
from sqlalchemy import text
from common import get_engine, ROOT
with get_engine().begin() as c:
    c.execute(text((ROOT / sys.argv[1]).read_text()))
print("ran", sys.argv[1])