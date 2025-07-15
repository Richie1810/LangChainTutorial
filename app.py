import streamlit as st
from openai import OpenAI
from dotenv import load_dotenv
import os
from qa_chain import load_and_build_qa_chain

# --- 初期設定 ---
load_dotenv()
# api_key = os.getenv("OPENAI_API_KEY")
api_key = st.secrets["OPENAI_API_KEY"]
client = OpenAI(api_key=api_key)

st.set_page_config(page_title="LangChain Tutorial", layout="centered")
st.title("OpenAIのAPIを使ってみる")

# --- タブ構成（IR質問 / 画像生成） ---
tab1, tab2 = st.tabs(["📄 IR資料に質問", "🎨 画像を生成"])

# --- IR QAタブ ---
with tab1:
    st.subheader("IR資料検索")
    st.markdown("アップロード済みのIR資料から質問に答えます。")

    if "qa" not in st.session_state:
        with st.spinner("IR資料を読み込んでいます..."):
            st.session_state.qa = load_and_build_qa_chain()

    query = st.text_input("質問を入力してください", placeholder="2024年度の営業利益は？")
    if st.button("検索実行", key="qa_search"):
        if query.strip():
            with st.spinner("回答生成中..."):
                result = st.session_state.qa.run(query)
                st.success("✅ 回答:")
                st.markdown(result)
        else:
            st.warning("質問を入力してください。")

# --- 画像生成タブ ---
with tab2:
    st.subheader("画像生成（DALL·E 3）")
    prompt = st.text_input("画像プロンプト（日本語OK）", placeholder="未来的なNTTデータのオフィスビル")

    if st.button("画像を生成", key="image_generate"):
        if prompt.strip():
            with st.spinner("画像生成中..."):
                try:
                    response = client.images.generate(
                        model="dall-e-3",
                        prompt=prompt,
                        size="1024x1024",
                        quality="standard",
                        n=1
                    )
                    image_url = response.data[0].url
                    st.image(image_url, caption="生成された画像", use_column_width=True)
                except Exception as e:
                    st.error(f"エラーが発生しました: {e}")
        else:
            st.warning("プロンプトを入力してください。")
