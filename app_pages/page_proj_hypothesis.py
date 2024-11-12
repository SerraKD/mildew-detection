import streamlit as st


def page_project_hypothesis_body():
    # https://treefruit.wsu.edu/crop-protection/disease-management/cherry-powdery-mildew/#:~:text=Powdery%20mildew%20of%20sweet%20and,1).

    st.write("## Project Hypothesis and Validation")

    st.info(
        f"Powdery mildew of sweet and sour cherry is caused by "
        f"Podosphaera clandestina, an obligate biotrophic fungus."
        f" Mid- and late-season sweet cherry (Prunus avium) cultivars"
        f" are commonly affected, rendering them unmarketable due to "
        f"the covering of white fungal growth on the cherry surface. \n\n"
        f"Initial symptoms, often occurring 7 to 10 days after the onset "
        f"of the first irrigation, are light roughly circular, "
        f"powdery-looking patches on young, susceptible leaves "
        f"(newly unfolded, and light green expanding leaves). "
        f"Older leaves develop an age-related (ontogenic) "
        f"resistance to powdery mildew and are naturally more resistant to "
        f"infection than younger leaves. Look for early leaf infections on "
        f"root suckers, the interior of the canopy, or the crotch of the"
        f" tree where humidity is high. \n\n"
        f" The disease is more likely to initiate on the "
        f"undersides (abaxial) of leaves but will occur"
        f" on both sides at later stages. As the season "
        f" progresses and infection is spread by wind, leaves may become "
        f"distorted, curling upward. Severe infections may cause leaves to "
        f"pucker and twist. Newly developed leaves on new shoots become "
        f"progressively smaller, are often pale, and may be distorted."
    )

    st.success(
        f" The Image Montage shows the apparent difference between "
        f"healthy and powdery mildew-infected leaves. \n\n"
        f"Average, Variability, and Difference Images confirmed "
        f"the hypothesis by showing color differences within the "
        f"center of each leaf image, yet there are no clear "
        f"patterns to identify them by shape. \n\n"
        f"ML pipeline performance evaluation shows that it"
        f" differentiates a healthy leaf from an infected "
        f"leaf with 99% accuracy."
    )
    
