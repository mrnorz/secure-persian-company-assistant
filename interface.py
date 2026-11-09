"""Presentation helpers. All dynamic text uses native, escaped Streamlit elements."""
import streamlit as st
import time
from pathlib import Path

ASSETS = Path(__file__).parent / 'static'


def render_stream(stream):
    """Stream Markdown without dataframe detection or per-token UI rerenders."""
    output = st.empty()
    pieces = []
    last = 0.0
    for piece in stream:
        pieces.append(piece)
        now = time.monotonic()
        if now - last >= 0.08:
            output.markdown(''.join(pieces) + ' ▍')
            last = now
    text = ''.join(pieces).strip() or 'مدل پاسخی تولید نکرد. دوباره تلاش کنید.'
    output.markdown(text)
    return text


def style():
    # RTL needs a small CSS rule; colors and widget styling use native theme tokens.
    st.html('''<style>
    [data-testid="stMain"], [data-testid="stSidebar"], [data-testid="stChatInput"] {direction:rtl;text-align:right}
    [data-testid="stMarkdownContainer"] p {line-height:1.95}
    [data-testid="stCode"], pre, code {direction:ltr;text-align:left}
    </style>''')


def sources(items):
    if not items:
        return
    with st.expander(f"بررسی منابع · {len(items)} گزیده", icon=":material/library_books:"):
        for i, item in enumerate(items, 1):
            with st.container(border=True):
                st.badge(f"منبع [{i}] · صفحهٔ {item['page']}",icon=':material/description:',color='blue')
                st.text(item['filename'])
                st.text(item['text'])


def welcome():
    with st.container(border=True):
        copy,art = st.columns([1.55,1],vertical_alignment='center')
        with copy:
            st.badge('همراه هوشمند تیم شما',color='blue',icon=':material/auto_awesome:')
            st.title('دانش شرکت،\nیک گفتگو نزدیک‌تر.')
            st.markdown('میان اسناد کمتر بگردید. پاسخ روشن، خلاصهٔ کاربردی و اقدام بعدی را با منبع پیدا کنید.')
            st.caption('از پرسش شما تا متن منبع، در یک فضای کاری یکپارچه.')
        with art:
            st.image(str(ASSETS/'knowledge.svg'),width='stretch',alt='تصویر انتزاعی پیوند اسناد با یک دستیار هوشمند')
    st.write('')
    cards=st.columns(3)
    for col,icon,title,description in zip(cards,
            ['search','summarize','checklist'],
            ['پاسخ مستند','مرور سریع','اقدام بعدی'],
            ['پرسش دقیق از اسناد منتخب','نکات اصلی از بخش‌های مختلف سند','استخراج تعهدها و مهلت‌ها']):
        with col:
            with st.container(border=True,height='stretch'):
                st.markdown(f'### :material/{icon}:')
                st.markdown(f'**{title}**')
                st.caption(description)


def message_actions(message,key):
    with st.container(horizontal=True):
        with st.popover('برداشت پاسخ',icon=':material/content_copy:'):
            st.caption('با دکمهٔ گوشهٔ کادر، متن را کپی کنید.')
            st.code(message['content'],language=None,wrap_lines=True)
        st.download_button('دریافت پاسخ',conversation_export([message]),f'answer-{key}.md',
            'text/markdown',icon=':material/download:',key=f'answer_download_{key}')
        st.feedback('thumbs',key=f'feedback_{key}')


def conversation_export(messages):
    sections=[]
    for message in messages:
        text=('## شما\n\n' if message['role']=='user' else '## دستیار\n\n') + message['content']
        if message.get('sources'):
            text+='\n\n### منابع\n\n'+'\n\n'.join(
                f"[{i}] {s['filename']} — صفحه {s['page']}\n\n{s['text']}"
                for i,s in enumerate(message['sources'],1))
        sections.append(text)
    return '\n\n---\n\n'.join(sections)
