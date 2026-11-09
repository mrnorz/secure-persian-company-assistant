"""UI regression tests use an isolated database and mocked generation service."""
import json
import os
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from streamlit.testing.v1 import AppTest


class Response:
    def __enter__(self): return self
    def __exit__(self,*args): pass
    def raise_for_status(self): pass
    def iter_lines(self):
        yield json.dumps({'response':'Test answer [1]'}).encode()


class AppTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.env=patch.dict(os.environ,{'APP_DATA_DIR':self.tmp.name})
        self.env.start()
        self.at=AppTest.from_file('app.py',default_timeout=15).run()
        path=Path(self.tmp.name)/'secure_assistant.sqlite3'
        conn=sqlite3.connect(path)
        with conn:
            conn.execute("INSERT INTO documents(id,filename,min_role,departments,created_by) VALUES(1,'public.pdf','employee','[]','admin')")
            conn.execute("INSERT INTO documents(id,filename,min_role,departments,created_by) VALUES(2,'secret.pdf','manager','[]','admin')")
            conn.executemany('INSERT INTO chunks(document_id,page,text,embedding) VALUES(?,?,?,?)',[(i,1,'Public evidence' if i==1 else 'Secret evidence',np.array([1.,0.],dtype=np.float32).tobytes()) for i in [1,2]])
        conn.close()

    def tearDown(self):
        self.env.stop()
        self.tmp.cleanup()

    def login(self,username='employee'):
        self.at.text_input[0].set_value(username)
        self.at.text_input[1].set_value('change-me-'+username)
        self.at.button[0].click().run()
        self.assertFalse(self.at.exception)

    def test_login_library_and_employee_scope(self):
        self.login()
        self.assertEqual(list(self.at.multiselect(key='chosen_docs').options),['public.pdf · #1'])
        self.at.radio(key='page_view').set_value('کتابخانهٔ اسناد').run()
        self.assertFalse(self.at.exception)
        self.assertTrue(any(t.value=='public.pdf' for t in self.at.text))
        self.assertFalse(any(t.value=='secret.pdf' for t in self.at.text))

    def test_summary_stream_and_switch_conversation(self):
        self.login()
        summary=next(b for b in self.at.button if b.label=='خلاصهٔ اسناد منتخب')
        with patch('requests.post',return_value=Response()) as post:
            summary.click().run()
        self.assertFalse(self.at.exception)
        self.assertTrue(post.call_args.kwargs['stream'])
        payload=post.call_args.kwargs['json']['prompt']
        self.assertIn('Public evidence',payload)
        self.assertNotIn('Secret evidence',payload)
        self.assertEqual(self.at.session_state['messages'][-1]['content'],'Test answer [1]')
        old=self.at.session_state['active_thread']
        self.at.button(key='new_conversation').click().run()
        self.assertFalse(self.at.exception)
        self.assertEqual(self.at.session_state['messages'],[])
        self.at.selectbox(key='thread_picker').set_value(old).run()
        self.assertFalse(self.at.exception)
        self.assertEqual(self.at.session_state['messages'][-1]['content'],'Test answer [1]')

    def test_library_summary_callback_sets_scope(self):
        self.login('manager')
        self.at.radio(key='page_view').set_value('کتابخانهٔ اسناد').run()
        with patch('requests.post',return_value=Response()) as post:
            self.at.button(key='summary_doc_2').click().run()
        self.assertFalse(self.at.exception)
        self.assertEqual(self.at.session_state['chosen_docs'],[2])
        self.assertIn('Secret evidence',post.call_args.kwargs['json']['prompt'])
        self.assertNotIn('Public evidence',post.call_args.kwargs['json']['prompt'])


if __name__=='__main__': unittest.main()
