import streamlit as st
from app_pages.multi_page import MultiPage

# Load pages scripts
from app_pages.page_proj_summary import page_summary_body
from app_pages.page_proj_hypothesis import page_project_hypothesis_body


app = MultiPage(app_name="Mildew Detection in Cherry Leaves")

# Add app pages using .add_page()
app.add_page("Project Summary", page_summary_body)
app.add_page("Project Hypothesis and Validation", page_project_hypothesis_body)

# Run the app
app.run()