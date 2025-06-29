import streamlit as st
import requests

API_URL = "http://localhost:8000"
MAX_IMAGES_PER_ROW = 5

def image_url(addr):
    return f"{API_URL}/image/{addr['folder_name']}/{addr['image_name']}"

st.set_page_config(
    page_title="Multimodal RAG Keyframe Viewer",
    page_icon="🎬",
    layout="wide"
)

st.markdown(
    "<h1 style='text-align:center;'>🎬 Multimodal RAG Keyframe Viewer</h1>",
    unsafe_allow_html=True,
)
st.markdown("---")

# --- Sidebar controls ---
with st.sidebar:
    st.header("🔍 Search Options")
    with st.expander("Set Search Parameters", expanded=True):
        query = st.text_input("Text Query", value="", help="Describe what you want to find (e.g., 'car driving at night').")
        top_k = st.selectbox("Top-K Results", [5, 10, 20, 50, 100], index=1)
        window = st.selectbox("Window Frames (context)", [1, 2, 5, 10, 20, 50], index=3)
        search = st.button("🔎 Search", type="primary", use_container_width=True)
    st.markdown("---")
    st.caption("Built with ❤️ using Streamlit and FastAPI")

# -- Handle search button and display progressive results --
if search and query.strip():
    st.session_state["progressive_results"] = []  # Store result groups as they arrive
    placeholder = st.empty()
    with st.spinner("Searching keyframes..."):
        # 1. Search for keyframes (address list)
        sr = {"queries": [query], "limit": top_k}
        r = requests.post(f"{API_URL}/search", json=sr)
        if r.status_code != 200:
            st.sidebar.error(f"Search error: {r.text}")
            st.stop()
        keyframes = r.json()
        if not keyframes:
            st.sidebar.warning("No search results found.")
            st.stop()
        # 2. For each keyframe, request its surroundings one-by-one and display immediately
        results_so_far = []
        for idx, keyframe in enumerate(keyframes):
            single_surrounding_req = {
                "addresses": [keyframe],  # one keyframe at a time
                "window": window
            }
            r2 = requests.post(f"{API_URL}/surroundings", json=single_surrounding_req)
            if r2.status_code != 200:
                st.sidebar.error(f"Surroundings error: {r2.text}")
                break
            # r2.json() returns a list of groups, but here it's a single group inside a list
            group = r2.json()[0]
            results_so_far.append(group)
            # Now render all so far
            with placeholder.container():
                st.success(
                    f"Showing {len(results_so_far)} of {len(keyframes)} result group(s) "
                    f"for **query:** `{query}` | **Top-K:** {top_k} | **Window:** {window}"
                )
                st.markdown("#### Keyframe Contexts")
                for gidx, group in enumerate(results_so_far):
                    if not group:
                        continue
                    st.markdown(f"<h5 style='margin-bottom:4px;'>Result {gidx+1}</h5>", unsafe_allow_html=True)
                    
                    # Show the center image as a big featured image
                    center_idx = len(group) // 2
                    center_addr = group[center_idx]
                    st.image(
                        image_url(center_addr),
                        use_container_width=True,
                        caption=f"**CENTER FRAME**<br>{center_addr['folder_name']}/{center_addr['image_name']}",
                        output_format="JPEG"
                    )
                    st.markdown("&nbsp;", unsafe_allow_html=True)  # Spacer
                    
                    # Show the other images in a grid (excluding the center image)
                    grid_group = group[:center_idx] + group[center_idx+1:]
                    for start in range(0, len(grid_group), MAX_IMAGES_PER_ROW):
                        chunk = grid_group[start:start + MAX_IMAGES_PER_ROW]
                        cols = st.columns(len(chunk))
                        for col, addr in zip(cols, chunk):
                            with col:
                                st.image(
                                    image_url(addr),
                                    use_container_width=True,
                                    caption=f"{addr['folder_name']}/{addr['image_name']}",
                                )
                    st.markdown("<hr style='margin:10px 0;'>", unsafe_allow_html=True)

else:
    st.info("Enter a query and click **Search** to see results.")
