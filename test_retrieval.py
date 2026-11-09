from pathlib import Path
import ast
import sqlite3
import tempfile
import unittest
from contextlib import contextmanager
import numpy as np
from types import SimpleNamespace
from workspace import rank_chunks, summary_sample

# Exercise app functions without executing the UI or downloading models.
source=ast.parse(Path('app.py').read_text(encoding='utf-8-sig'))
names={'retrieve','normalize_fa','authorized_documents'}
namespace={'np':np,'json':__import__('json'),'re':__import__('re'),'unicodedata':__import__('unicodedata'),'TOP_K':4,'EMBED_MODEL':'test','rank_chunks':rank_chunks,'summary_sample':summary_sample}
exec(compile(ast.Module(body=[n for n in source.body if isinstance(n,ast.FunctionDef) and n.name in names],type_ignores=[]),'app.py','exec'),namespace)

class RetrievalTest(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(dir='.')
        self.path=Path(self.tmp.name)/'test.sqlite'
        class State(dict):
            def __getattr__(self,name): return self[name]
            def __setattr__(self,name,value): self[name]=value
        namespace['st']=SimpleNamespace(session_state=State())
        namespace['DB']=self.path
        @contextmanager
        def db():
            c=sqlite3.connect(self.path); c.row_factory=sqlite3.Row
            try:
                with c: yield c
            finally: c.close()
        self.db=db
        namespace['db']=db
        class Encoder:
            def get_sentence_embedding_dimension(self): return 2
            def encode(self,texts,**kwargs): return np.asarray([[1.,0.] for _ in texts],dtype=np.float32)
        namespace['model']=lambda:Encoder()
        with db() as c:
            c.executescript('CREATE TABLE documents(id,filename,min_role,departments); CREATE TABLE chunks(id,document_id,page,text,embedding);')
            c.executemany('INSERT INTO documents VALUES (?,?,?,?)',[(1,'public.pdf','employee','[]'),(2,'secret.pdf','manager','[]'),(3,'sales.pdf','employee','["sales"]'),(4,'salesops.pdf','employee','["salesops"]')])
            c.executemany('INSERT INTO chunks VALUES (?,?,?,?,?)',[(i,i,1,'text'+str(i),np.array([1.,0.],dtype=np.float32).tobytes()) for i in range(1,5)])
    def tearDown(self): self.tmp.cleanup()
    def test_employee_cannot_retrieve_manager_or_partial_department(self):
        result=namespace['retrieve']('test',{'username':'staff','role':'employee','department':'sales'},document_ids=[1,2,3,4])
        self.assertEqual({r['filename'] for r in result},{'public.pdf','sales.pdf'})
    def test_summary_obeys_selected_document_without_embedding(self):
        namespace['model']=lambda: (_ for _ in ()).throw(AssertionError('summary loaded embedding'))
        result=namespace['retrieve']('خلاصه',{'username':'manager','role':'manager','department':''},document_ids=[2],summary=True)
        self.assertEqual([r['filename'] for r in result],['secret.pdf'])
    def test_mixed_dimensions_repaired(self):
        with self.db() as c: c.execute('UPDATE chunks SET embedding=? WHERE id=1',(np.array([1.,0.,0.],dtype=np.float32).tobytes(),))
        result=namespace['retrieve']('test',{'username':'staff','role':'employee','department':'sales'},document_ids=[1])
        self.assertEqual(len(result),1)
        with self.db() as c: self.assertEqual(c.execute('SELECT length(embedding) FROM chunks WHERE id=1').fetchone()[0],8)
    def test_empty_selection(self):
        self.assertEqual(namespace['retrieve']('test',{'username':'staff','role':'employee','department':'sales'},document_ids=[]),[])

    def test_permission_revocation_ignores_cached_index(self):
        user={'username':'staff','role':'employee','department':'sales'}
        self.assertEqual(len(namespace['retrieve']('test',user,document_ids=[1])),1)
        with self.db() as c: c.execute("UPDATE documents SET min_role='manager' WHERE id=1")
        self.assertEqual(namespace['retrieve']('test',user,document_ids=[1]),[])

    def test_query_embedding_is_reused(self):
        calls=[]
        class Encoder:
            def get_sentence_embedding_dimension(self): return 2
            def encode(self,texts,**kwargs):
                calls.append(texts)
                return np.array([[1.,0.] for _ in texts],dtype=np.float32)
        encoder=Encoder()
        namespace['model']=lambda:encoder
        user={'username':'staff','role':'employee','department':'sales'}
        namespace['retrieve']('test',user,document_ids=[1])
        namespace['retrieve']('test',user,document_ids=[1])
        self.assertEqual(len(calls),1)

if __name__=='__main__': unittest.main()
