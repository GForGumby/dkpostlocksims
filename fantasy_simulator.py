import streamlit as st
import pandas as pd
import numpy as np
from numba import jit
import io

@jit(nopython=True)
def generate_projection(median, std_dev):
    fluctuation = np.random.uniform(-0.01, 0.01) * median
    return max(0, np.random.normal(median, std_dev) + fluctuation)

@jit(nopython=True)
def get_payout(rank):
    if rank == 1:
        return 30000.00
    elif rank == 2:
        return 15000.00
    elif rank == 3:
        return 7500.00
    elif rank == 4:
        return 6000.00
    elif rank == 5:
        return 5000.00
    elif rank == 6:
        return 4250.00
    elif rank == 7:
        return 3750.00
    elif rank == 8:
        return 3500.00
    elif rank == 9:
        return 3250.00
    elif rank == 10:
        return 3000.00
    elif 11 <= rank <= 25:
        return 1000.00
    elif 26 <= rank <= 50:
        return 500.00
    elif 51 <= rank <= 100:
        return 125.00
    elif 101 <= rank <= 200:
        return 55.00
    elif 201 <= rank <= 500:
        return 35.00
    elif 501 <= rank <= 1000:
        return 25.00
    elif 1001 <= rank <= 2000:
        return 20.00
    elif 2001 <= rank <= 3000:
        return 15.00
    elif 3001 <= rank <= 7500:
        return 12.00
    elif 7501 <= rank <= 14250:
        return 10.00
    else:
        return 0.00

def prepare_draft_results(draft_results_df):
    teams = draft_results_df['Team'].unique()
    num_teams = len(teams)
    draft_results = np.empty((num_teams, 9), dtype='U50')

    for idx, team in enumerate(teams):
        team_row = draft_results_df[draft_results_df['Team'] == team].iloc[0]
        for i in range(9):
            column_name = f'G{i}' if i > 0 else 'G'
            draft_results[idx, i] = team_row[column_name]

    return draft_results, teams

def simulate_team_projections(draft_results, projection_lookup, num_simulations):
    num_teams = draft_results.shape[0]
    total_payouts = np.zeros(num_teams)

    for sim in range(num_simulations):
        total_points = np.zeros(num_teams)
        for i in range(num_teams):
            for j in range(9):  # Loop through all 9 players
                player_name = draft_results[i, j]
                if player_name in projection_lookup:
                    proj, projsd = projection_lookup[player_name]
                    simulated_points = generate_projection(proj, projsd)
                    total_points[i] += simulated_points

        # Rank teams
        ranks = total_points.argsort()[::-1].argsort() + 1

        # Assign payouts and accumulate them
        payouts = np.array([get_payout(rank) for rank in ranks])
        total_payouts += payouts

    # Calculate average payout per team
    avg_payouts = total_payouts / num_simulations
    return avg_payouts

def run_parallel_simulations(num_simulations, draft_results_df, projection_lookup):
    draft_results, teams = prepare_draft_results(draft_results_df)
    avg_payouts = simulate_team_projections(draft_results, projection_lookup, num_simulations)
    
    # Prepare final results
    final_results = pd.DataFrame({
        'Team': teams,
        'Average_Payout': avg_payouts
    })
    
    return final_results

def main():
    st.title("Fantasy Contest Simulator")

    # File upload for projections
    st.header("Upload Projections")
    projections_file = st.file_uploader("Upload projections CSV file", type="csv")
    
    if projections_file is not None:
        projections_df = pd.read_csv(projections_file)
        st.write("Projections data loaded successfully!")
        st.dataframe(projections_df.head())

        # File upload for draft results
        st.header("Upload Draft Results")
        draft_results_file = st.file_uploader("Upload draft results CSV file", type="csv")
        
        if draft_results_file is not None:
            draft_results_df = pd.read_csv(draft_results_file)
            st.write("Draft results data loaded successfully!")
            st.dataframe(draft_results_df.head())

            # Create projection lookup dictionary
            projection_lookup = {
                row['Player']: (row['Projection'], row['StdDev'])
                for _, row in projections_df.iterrows()
            }

            # Run simulations
            st.header("Run Simulations")
            num_simulations = st.slider("Number of simulations", 1000, 100000, 10000, 1000)
            
            if st.button("Run Simulations"):
                with st.spinner("Running simulations..."):
                    final_results = run_parallel_simulations(num_simulations, draft_results_df, projection_lookup)
                
                st.success("Simulations completed!")
                st.dataframe(final_results)

                # Download results
                csv = final_results.to_csv(index=False)
                st.download_button(
                    label="Download Results CSV",
                    data=csv,
                    file_name="simulation_results.csv",
                    mime="text/csv",
                )

if __name__ == "__main__":
    main()
