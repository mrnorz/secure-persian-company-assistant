import unittest
import numpy as np
from workspace import contextual_query, pack_context, rank_chunks, summary_sample, ensure_threads, add_thread


class WorkspaceTests(unittest.TestCase):
    def row(self,doc,text,page=1):
        return {'document_id':doc,'filename':f'{doc}.pdf','page':page,'text':text,'embedding':np.array([1.,0.],dtype=np.float32).tobytes()}

    def test_hybrid_prefers_exact_term_and_removes_duplicate_text(self):
        rows=[self.row(1,'annual leave'),self.row(2,'insurance deadline'),self.row(3,'insurance deadline')]
        ranked=rank_chunks('insurance',rows,[1.,0.],4)
        self.assertEqual(ranked[0]['text'],'insurance deadline')
        self.assertEqual(len(ranked),2)

    def test_summary_covers_documents_before_extra_pages(self):
        rows=[self.row(1,str(i),i) for i in range(20)]+[self.row(2,'other')]
        chosen=summary_sample(rows,4)
        self.assertEqual({r['document_id'] for r in chosen[:2]},{1,2})

    def test_context_never_exceeds_budget_including_headers(self):
        rows=[self.row(i,'intro '*100+'target '+'tail '*100) for i in range(4)]
        packed=pack_context(rows,1200,'target')
        self.assertLessEqual(sum(len(r['text'])+len(r['filename'])+60 for r in packed),1200)
        self.assertTrue(all('target' in r['text'] for r in packed))

    def test_followup_context_is_scoped_to_selected_documents(self):
        messages=[{'role':'user','content':'Insurance coverage','document_ids':[1]}]
        self.assertIn('Insurance coverage',contextual_query('deadline?',messages,[1]))
        self.assertNotIn('Insurance coverage',contextual_query('deadline?',messages,[2]))

    def test_permission_loss_clears_all_threads_and_caches(self):
        state={}
        ensure_threads(state,('staff','employee','sales',((1,'employee','[]'),(2,'employee','[]'))))
        state['messages'].append({'role':'assistant','content':'private'})
        state['query_cache']={'private':1}
        ensure_threads(state,('staff','employee','sales',((1,'employee','[]'),)))
        self.assertEqual(state['messages'],[])
        self.assertNotIn('query_cache',state)

    def test_switch_thread_keeps_messages_and_new_document_keeps_history(self):
        state={}
        key=('staff','employee','sales',((1,'employee','[]'),))
        ensure_threads(state,key)
        original=state['active_thread']
        state['messages'].append({'role':'user','content':'question'})
        add_thread(state)
        self.assertEqual(state['messages'],[])
        state['active_thread']=original
        ensure_threads(state,('staff','employee','sales',((1,'employee','[]'),(2,'employee','[]'))))
        self.assertEqual(state['messages'][0]['content'],'question')


if __name__=='__main__': unittest.main()
