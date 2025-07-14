import streamlit as st
import configparser
import os
import time

def run():
    st.title("Optimisation Page")
    # === SETTINGS FILE ===
    st.subheader("Loaded Configuration")
    config = configparser.ConfigParser()
    #cfg_path = "settings.cfg"
    cfg_path = "/mount/src/foundationcalc/structural_app2/settings.cfg"

    if os.path.exists(cfg_path):
        config.read(cfg_path)
        template = config["DEFAULT"].get("TEMPLATE", "Not found")
        sheet = config["DEFAULT"].get("SHEET", "Not found")

        st.write(f"**Template File:** `{template}`")
        st.write(f"**Sheet Name:** `{sheet}`")
    else:
        st.error(f"`{cfg_path}` file not found.")

    # === INPUT LAYOUT ===
    st.subheader("Optimisation Parameters")
    col1, col2, col3 = st.columns(3)

    with col1:
        q_max = st.number_input("Maximum Bearing Pressure (q_max)", value=150.0)
        X_min = st.number_input("Min X dimension (X_min)", value=1.0)
        X_max = st.number_input("Max X dimension (X_max)", value=150.0)

    with col2:
        Z_min = st.number_input("Min Z dimension (Z_min)", value=0.75)
        Z_max = st.number_input("Max Z dimension (Z_max)", value=150.0)
        Y_min = st.number_input("Min Y dimension (Y_min)", value=1.0)

    with col3:
        Y_max = st.number_input("Max Y dimension (Y_max)", value=150.0)
        pQ = st.number_input("Load factor (pQ)", value=1.0)
        FoS = st.number_input("Factor of Safety (FoS)", value=1.25)

    u = st.number_input("Friction Coefficient (u)", value=0.45)
    Square = st.checkbox("Use square optimisation (otherwise rectangular)", value=True)

# === SESSION STATE INIT ===
    if "optimisation_started" not in st.session_state:
        st.session_state.optimisation_started = False

    # === PLACEHOLDER TO MANAGE BUTTON/DISPLAY ===
    action_area = st.empty()

    # === BUTTON LOGIC ===
    if not st.session_state.optimisation_started:
        with action_area.container():
            if st.button("Confirm and Proceed"):
                st.session_state.optimisation_started = True
                st.rerun()

    else:
        with action_area.container():
        #with st.spinner("Optimising..."):
            info_area = st.empty()
            progress_area = st.empty()
            error_area = st.empty()
            #error_area.error("No Errors")
            info_area.info("✅ Optimisation Started")
            time.sleep(1)
            ###########################################################################################################################    
            ###########################################################################################################################
            ###########################################################################################################################
            ###########################################################################################################################
            ### TOM's Code
            # Import packages
            import pandas as pd
            from gekko import GEKKO
            import numpy as np
            import sys
            ###################################################################################################################
            # File paths and global variables
            #file_path = r'C:\Users\GDP\OneDrive - Tony Gee and Partners LLP\Documents\Automation\__2025__Digital__\DEV - WIP\P&E Streamlit Toms App\structural_app2\loads_template'
            file_path = "/mount/src/foundationcalc/structural_app2/loads_template/"
            #loads_file_path = r'C:\Users\GDP\OneDrive - Tony Gee and Partners LLP\Documents\Automation\__2025__Digital__\DEV - WIP\P&E Streamlit Toms App\Rev01\foundation_loadsv2.xlsx'
            loads_file_path = os.path.join(file_path, template)
            #optimised_output_path = r'C:\Users\GDP\OneDrive - Tony Gee and Partners LLP\Documents\Automation\__2025__Digital__\DEV - WIP\P&E Streamlit Toms App\structural_app2\output\optimised_result.xlsx'
            optimised_output_path = "/mount/src/foundationcalc/structural_app2/output/optimised_result.xlsx"
            #loads_sheet_name = 'Loading Summary'
            loads_sheet_name = sheet
            ###################################################################################################################
            # Import excel into pandas dataframe
            # Load data from the specific sheet into a DataFrame
            table = pd.read_excel(loads_file_path, sheet_name=loads_sheet_name)

            df_simplified=table
            ###################################################################################################################
            ###################################################################################################################
            ### Analysis function 1
            def foundation_inside_middle_third_square_df(loadcases_df, pQ, q_max, Z_min, Z_max, X_min, X_max):
                """
                Optimizes square foundation dimensions (X, X, Z) for multiple load cases.
                
                Parameters:
                    loadcases_df (pd.DataFrame): DataFrame containing load case parameters with specific column names.
                    pQ, q_max, Z_min, Z_max, X_min, X_max: Optimization bounds and parameters.
                
                Returns:
                    pd.DataFrame: DataFrame with optimized dimensions (X, X, Z) and calculated e_x for each load case.
                """
                results = []
                try:
                    for _, row in loadcases_df.iterrows():
                        # Initialize Gekko model
                        m = GEKKO(remote=False)
                        m.options.SOLVER = 3  # Use IPOPT solver

                        # Define variables for foundation dimensions
                        X = m.Var(value=1, lb=X_min, ub=X_max)  # Foundation breadth (square)
                        Z = m.Var(value=10, lb=Z_min, ub=Z_max)  # Foundation depth

                        # Wrap q_max in a GEKKO parameter
                        q_max_param = m.Param(value=q_max)

                        # Wrap load case parameters in GEKKO parameters
                        Fz = m.Param(value=row['Vertical Fz (kN)'])
                        Fx = m.Param(value=row['⊥ to Span, Fx (kN)'])
                        Fy = m.Param(value=row['|| to Span, Fy (kN)'])
                        Mx = m.Param(value=row['Moment about ⊥, Mx (kNm)'])
                        My = m.Param(value=row['Moment about ||, My (kNm)'])
                        name = row.get('Name', None)  # Optional: Get the 'Name' column if it exist


                        # Calculate effective foundation weight
                        effective_Fz = m.Intermediate(Fz + (X * X * Z * 24))

                        # Bearing pressure constraints for the current load case
                        term_1 = m.Intermediate(effective_Fz / (X * X))
                        term_2 = m.Intermediate(6 * effective_Fz * ((My + Fx * Z) / effective_Fz) / (X**3))
                        term_3 = m.Intermediate(6 * effective_Fz * ((Mx + Fy * Z) / effective_Fz) / (X**3))
                        
                        m.Equation(term_1 + term_2 + term_3 < q_max_param)  # Maximum bearing pressure
                        m.Equation(term_1 + term_2 + term_3 > 0)            # Minimum bearing pressure
                        
                        # Middle-third requirements for the current load case
                        ex = m.Intermediate((My + Fx * Z) / effective_Fz)
                        ey = m.Intermediate((Mx + Fy * Z) / effective_Fz)
                        m.Equation((X / 3) > ex)
                        m.Equation((X / 3) > ey)

                        # Objective: Minimize foundation volume (square) X * X * Z
                        m.Minimize(X * X * Z)

                        # Solve model
                        m.options.IMODE = 3  # Steady state optimization
                        m.solve(disp=False)

                        # Extract results
                        X_opt = X.value[0]
                        Z_opt = Z.value[0]

                        # Calculate e_x
                        e_x = 2 * ex.value[0]

                        # Append results for this load case
                        results.append({
                            'Name': name,  # Include the Name column or other identifier
                            'X': round(X_opt,2),
                            'X2': round(X_opt,2),  # Assuming square foundation, so X = X2
                            'Z': round(Z_opt,2),
                            'e_x': round(e_x,2)
                        })

                except Exception as e:
                    print(f"Error in optimization: {e}")
                    #progress_area.info(f"Error in optimization: {e}")
                    error_area.error(f"Error in optimization: {e}")
                    return pd.DataFrame(columns=['X', 'X2', 'Z', 'e_x'])

                # Convert results to a DataFrame
                results_df = pd.DataFrame(results)

                return results_df

            ###################################################################################################################
            ###################################################################################################################
            ### Analysis function 2
            def foundation_inside_middle_third_rect_df(loadcases_df, pQ, q_max, Z_min, Z_max, X_min, X_max, Y_min, Y_max):
                """
                Optimizes square foundation dimensions (X, X, Z) for multiple load cases.
                
                Parameters:
                    loadcases_df (pd.DataFrame): DataFrame containing load case parameters with specific column names.
                    pQ, q_max, Z_min, Z_max, X_min, X_max: Optimization bounds and parameters.
                
                Returns:
                    pd.DataFrame: DataFrame with optimized dimensions (X, X, Z) and calculated e_x for each load case.
                """
                results = []

                try:
                    for _, row in loadcases_df.iterrows():
                        # Initialize Gekko model
                        m = GEKKO(remote=False)
                        m.options.SOLVER = 3  # Use IPOPT solver

                        # Define variables for foundation dimensions
                        Y = m.Var(value=1, lb=Y_min, ub=Y_max)  # Foundation breadth (square)
                        X = m.Var(value=1, lb=X_min, ub=X_max)  # Foundation breadth (square)
                        Z = m.Var(value=10, lb=Z_min, ub=Z_max)  # Foundation depth

                        # Wrap q_max in a GEKKO parameter
                        q_max_param = m.Param(value=q_max)

                        # Wrap load case parameters in GEKKO parameters
                        Fz = m.Param(value=row['Vertical Fz (kN)'])
                        Fx = m.Param(value=row['⊥ to Span, Fx (kN)'])
                        Fy = m.Param(value=row['|| to Span, Fy (kN)'])
                        Mx = m.Param(value=row['Moment about ⊥, Mx (kNm)'])
                        My = m.Param(value=row['Moment about ||, My (kNm)'])
                        name = row.get('Name', None)  # Optional: Get the 'Name' column if it exist


                        # Calculate effective foundation weight
                        effective_Fz = m.Intermediate(Fz + (X * Y * Z * 24))

                        # Bearing pressure constraints for the current load case
                        term_1 = m.Intermediate(effective_Fz / (X * Y))
                        term_2 = m.Intermediate(6 * effective_Fz * ((My + Fx * Z) / effective_Fz) / (X**3))
                        term_3 = m.Intermediate(6 * effective_Fz * ((Mx + Fy * Z) / effective_Fz) / (Y**3))
                        
                        m.Equation(term_1 + term_2 + term_3 < q_max_param)  # Maximum bearing pressure
                        m.Equation(term_1 + term_2 + term_3 > 0)            # Minimum bearing pressure
                        
                        # Middle-third requirements for the current load case
                        ex = m.Intermediate((My + Fx * Z) / effective_Fz)
                        ey = m.Intermediate((Mx + Fy * Z) / effective_Fz)
                        m.Equation((X / 3) > ex)
                        m.Equation((Y / 3) > ey)

                        # Objective: Minimize foundation volume (square) X * X * Z
                        m.Minimize(X * Y * Z)

                        # Solve model
                        m.options.IMODE = 3  # Steady state optimization
                        m.solve(disp=False)

                        # Extract results
                        Y_opt = Y.value[0]
                        X_opt = X.value[0]
                        Z_opt = Z.value[0]

                        # Calculate e_x
                        e_x = 2 * ex.value[0]
                        e_y = 2 * ey.value[0]


                        # Append results for this load case
                        results.append({
                            'Name': name,  # Include the Name column or other identifier
                            'X': round(X_opt,2),
                            'Y': round(Y_opt,2),  # Assuming square foundation, so X = X2
                            'Z': round(Z_opt,2),
                            'e_x': round(e_x,2),
                            'e_y': round(e_y,2)

                        })

                except Exception as e:
                    print(f"Error in optimization: {e}")
                    #progress_area.info(f"Error in optimization: {e}")
                    error_area.error(f"Error in optimization: {e}")
                    return pd.DataFrame(columns=['X', 'Y', 'Z', 'e_x','e_y'])

                # Convert results to a DataFrame
                results_df = pd.DataFrame(results)

                return results_df
            ###################################################################################################################
            ###################################################################################################################
            ### Analysis function 3
            def foundation_outside_middle_third_square_df(loadcases_df, q_max, Z_min, Z_max, X_min, X_max):
                """
                Optimizes square foundation dimensions (X, X, Z) for multiple load cases.
                
                Parameters:
                    loadcases_df (pd.DataFrame): DataFrame containing load case parameters.
                    q_max, Z_min, Z_max, X_min, X_max: Optimization bounds and parameters.
                
                Returns:
                    pd.DataFrame: DataFrame with optimized foundation dimensions (X, X, Z) for each load case and calculated e_x.
                """
                try:
                    results = []

                    for _, row in loadcases_df.iterrows():
                        # Extract load case parameters
                        Fz = row['Vertical Fz (kN)']
                        Fx = row['⊥ to Span, Fx (kN)']
                        Fy = row['|| to Span, Fy (kN)']
                        Mx = row['Moment about ⊥, Mx (kNm)']
                        My = row['Moment about ||, My (kNm)']
                        name = row.get('Name', None)  # Optional: Get the 'Name' column if it exist

                        # Initialize GEKKO model
                        m = GEKKO(remote=False)
                        m.options.SOLVER = 3  # Use IPOPT solver

                        # Define variables for foundation dimensions
                        X = m.Var(value=100, lb=X_min, ub=X_max)  # Foundation breadth (square)
                        Z = m.Var(value=0.75, lb=Z_min, ub=Z_max)  # Foundation depth

                        # Effective foundation weight
                        effective_Fz = m.Intermediate(Fz + (X * X * Z * 24))

                        # Constraints based on bearing pressure and eccentricity
                        m.Equation(((4 * effective_Fz) / (3 * (X - 2 * ((My + Fx * Z) / effective_Fz)) * (X - 2 * ((Mx + Fy * Z) / effective_Fz)))) < q_max)
                        m.Equation(abs(2 * ((My + Fx * Z) / effective_Fz)) < X)
                        m.Equation(abs(2 * ((Mx + Fy * Z) / effective_Fz)) < X)

                        # Objective: Minimize foundation volume
                        m.Minimize(X * X * Z)

                        # Solve optimization
                        m.options.IMODE = 3  # Steady state optimization
                        m.solve(disp=False)

                        # Extract optimized results
                        X_opt = X.value[0]
                        Z_opt = Z.value[0]

                        # Calculate e_x based on the optimized values of X_opt, Z_opt
                        e_x = 2 * ((Mx + Fy * Z_opt) / (Fz + (X_opt * X_opt * Z_opt * 24)))

                        # Append the result to the results list
                        results.append({
                            'Name': name,  # Include the Name column or other identifier
                            'X': round(X_opt,2),
                            'X2': round(X_opt,2),  # Assuming square foundation, so X = X2
                            'Z': round(Z_opt,2),
                            'e_x': round(e_x,2)  # Add e_x to the results
                        })

                except Exception as e:
                    print(f"Error in optimization: {e}")
                    #progress_area.info(f"Error in optimization: {e}")
                    error_area.error(f"Error in optimization: {e}")
                    
                    return pd.DataFrame(columns=['X', 'X2', 'Z', 'e_x'])

                # Convert results to a DataFrame
                results_df = pd.DataFrame(results)

                return results_df
            ###################################################################################################################
            ###################################################################################################################
            ### Analysis function 4
            def foundation_outside_middle_third_rect_df(loadcases_df, q_max, Z_min, Z_max, X_min, X_max, Y_min, Y_max):
                """
                Optimizes square foundation dimensions (X, X, Z) for multiple load cases.
                
                Parameters:
                    loadcases_df (pd.DataFrame): DataFrame containing load case parameters.
                    q_max, Z_min, Z_max, X_min, X_max: Optimization bounds and parameters.
                
                Returns:
                    pd.DataFrame: DataFrame with optimized foundation dimensions (X, X, Z) for each load case and calculated e_x.
                """
                try:
                    results = []

                    for _, row in loadcases_df.iterrows():
                        # Extract load case parameters
                        Fz = row['Vertical Fz (kN)']
                        Fx = row['⊥ to Span, Fx (kN)']
                        Fy = row['|| to Span, Fy (kN)']
                        Mx = row['Moment about ⊥, Mx (kNm)']
                        My = row['Moment about ||, My (kNm)']
                        name = row.get('Name', None)  # Optional: Get the 'Name' column if it exist

                        # Initialize GEKKO model
                        m = GEKKO(remote=False)
                        m.options.SOLVER = 3  # Use IPOPT solver

                        # Define variables for foundation dimensions
                        X = m.Var(value=100, lb=X_min, ub=X_max)  # Foundation breadth (square)
                        Y = m.Var(value=100, lb=Y_min, ub=Y_max)  # Foundation breadth (square)

                        Z = m.Var(value=0.75, lb=Z_min, ub=Z_max)  # Foundation depth

                        # Effective foundation weight
                        effective_Fz = m.Intermediate(Fz + (X * Y * Z * 24))

                        # Constraints based on bearing pressure and eccentricity
                        m.Equation(((4 * effective_Fz) / (3 * (X - 2 * ((My + Fx * Z) / effective_Fz)) * (Y - 2 * ((Mx + Fy * Z) / effective_Fz)))) < q_max)
                        m.Equation(abs(2 * ((My + Fx * Z) / effective_Fz)) < X)
                        m.Equation(abs(2 * ((Mx + Fy * Z) / effective_Fz)) < Y)

                        # Objective: Minimize foundation volume
                        m.Minimize(X * Y * Z)

                        # Solve optimization
                        m.options.IMODE = 3  # Steady state optimization
                        m.solve(disp=False)

                        # Extract optimized results
                        X_opt = X.value[0]
                        Y_opt = Y.value[0]
                        Z_opt = Z.value[0]

                        # Calculate e_x based on the optimized values of X_opt, Z_opt
                        e_x = 2 * ((My + Fx * Z_opt) / (Fz + (X_opt * Y_opt * Z_opt * 24)))
                        e_y = 2 * ((Mx + Fy * Z_opt) / (Fz + (X_opt * Y_opt * Z_opt * 24)))


                        # Append the result to the results list
                        results.append({
                            'Name': name,  # Include the Name column or other identifier
                            'X': round(X_opt,2),
                            'Y': round(Y_opt,2),  # Assuming square foundation, so X = X2
                            'Z': round(Z_opt,2),
                            'e_x': round(e_x,2),  # Add e_x to the results
                            'e_y': round(e_y,2)  # Add e_x to the results

                        })

                except Exception as e:
                    print(f"Error in optimization: {e}")
                    #progress_area.info("Error in optimization: {e}")
                    error_area.error(f"Error in optimization: {e}")
                    return pd.DataFrame(columns=['X', 'Y', 'Z', 'e_x','e_y'])

                # Convert results to a DataFrame
                results_df = pd.DataFrame(results)

                return results_df
            ###################################################################################################################
            ###################################################################################################################
            ### Optimize function 1
            def optimize_foundations_square(loadcases_df, pQ, q_max, Z_min, Z_max, X_min, X_max):
                """
                Applies both inside and outside middle third optimizations for load cases grouped by 'Name'.

                Parameters:
                    loadcases_df (pd.DataFrame): DataFrame containing load case parameters.
                    pQ, q_max, Z_min, Z_max, X_min, X_max: Optimization bounds and parameters.

                Returns:
                    pd.DataFrame: DataFrame containing summarized results for each foundation.
                """
                all_results = []

                # Group the load cases by 'Name'
                grouped = loadcases_df.groupby('Name')

                for name, group in grouped:
                    print(f"Processing load cases for {name}...")
                    progress_area.info(f"Processing load cases for {name}...")

                    # Initialize result placeholders
                    result_inside = {'X': np.nan, 'X2': np.nan, 'Z': np.nan, 'e_x': np.nan}
                    result_outside = {'X': np.nan, 'X2': np.nan, 'Z': np.nan, 'e_x': np.nan}

                    # Run inside middle third optimization
                    try:
                        results_inside = foundation_inside_middle_third_square_df(group, pQ, q_max, Z_min, Z_max, X_min, X_max)
                        if not results_inside.empty:
                            max_X_row_inside = results_inside.loc[results_inside['X'].idxmax()]
                            result_inside = {
                                'X': round(max_X_row_inside['X'], 2),
                                'X2': round(max_X_row_inside['X2'], 2),
                                'Z': round(max_X_row_inside['Z'], 2),
                                'e_x': round(max_X_row_inside['e_x'], 2)
                            }
                        else:
                            print(f"No feasible solution for inside middle third optimization for {name}.")
                            progress_area.info(f"No feasible solution for inside middle third optimization for {name}.")
                    except Exception as e:
                        print(f"Error in inside middle third optimization for {name}: {e}")
                        #progress_area.info(f"Error in inside middle third optimization for {name}: {e}")
                        error_area.error(f"Error in inside middle third optimization for {name}: {e}")

                    # Run outside middle third optimization
                    try:
                        results_outside = foundation_outside_middle_third_square_df(group, q_max, Z_min, Z_max, X_min, X_max)
                        if not results_outside.empty:
                            max_X_row_outside = results_outside.loc[results_outside['X'].idxmax()]
                            result_outside = {
                                'X': round(max_X_row_outside['X'], 2),
                                'X2': round(max_X_row_outside['X2'], 2),
                                'Z': round(max_X_row_outside['Z'], 2),
                                'e_x': round(max_X_row_outside['e_x'], 2)
                            }
                        else:
                            print(f"No feasible solution for outside middle third optimization for {name}.")
                            #progress_area.info(f"No feasible solution for outside middle third optimization for {name}.")
                            error_area.error(f"No feasible solution for outside middle third optimization for {name}.")
                    except Exception as e:
                        print(f"Error in outside middle third optimization for {name}: {e}")
                        #progress_area.info(f"Error in outside middle third optimization for {name}: {e}")
                        error_area.error(f"Error in outside middle third optimization for {name}: {e}")

                    # Append the combined results
                    all_results.append({
                        'Name': name,
                        # Inside middle third results
                        'X_in': result_inside['X'],
                        'X2_in': result_inside['X2'],
                        'Z_in': result_inside['Z'],
                        'e_x_in': result_inside['e_x'],
                        # Outside middle third results
                        'X_out': result_outside['X'],
                        'X2_out': result_outside['X2'],
                        'Z_out': result_outside['Z'],
                        'e_x_out': result_outside['e_x']
                    })

                # Convert all results into a DataFrame
                final_results_df = pd.DataFrame(all_results)

                return final_results_df
            ###################################################################################################################
            ###################################################################################################################
            ### Optimize function 2
            def optimize_foundations_rect(loadcases_df, pQ, q_max, Z_min, Z_max, X_min, X_max, Y_min, Y_max):
                """
                Applies both inside and outside middle third optimizations for load cases grouped by 'Name'.

                Parameters:
                    loadcases_df (pd.DataFrame): DataFrame containing load case parameters.
                    pQ, q_max, Z_min, Z_max, X_min, X_max: Optimization bounds and parameters.

                Returns:
                    pd.DataFrame: DataFrame containing summarized results for each foundation.
                """
                all_results = []

                # Group the load cases by 'Name'
                grouped = loadcases_df.groupby('Name')

                for name, group in grouped:
                    print(f"Processing load cases for {name}...")
                    progress_area.info(f"Processing load cases for {name}...")

                    # Initialize result placeholders
                    result_inside = {'X': np.nan, 'Y': np.nan, 'Z': np.nan, 'e_x': np.nan, 'e_y': np.nan}
                    result_outside = {'X': np.nan, 'Y': np.nan, 'Z': np.nan, 'e_x': np.nan, 'e_y': np.nan}

                    # Run inside middle third optimization
                    try:
                        results_inside = foundation_inside_middle_third_rect_df(group, pQ, q_max, Z_min, Z_max, X_min, X_max, Y_min, Y_max)
                        if not results_inside.empty:
                            max_X_row_inside = results_inside.loc[results_inside['X'].idxmax()]
                            result_inside = {
                                'X': round(max_X_row_inside['X'], 2),
                                'Y': round(max_X_row_inside['Y'], 2),
                                'Z': round(max_X_row_inside['Z'], 2),
                                'e_x': round(max_X_row_inside['e_x'], 2),
                                'e_y': round(max_X_row_inside['e_y'], 2)

                            }
                        else:
                            print(f"No feasible solution for inside middle third optimization for {name}.")
                            progress_area.info(f"No feasible solution for inside middle third optimization for {name}.")
                    except Exception as e:
                        print(f"Error in inside middle third optimization for {name}: {e}")
                        #progress_area.info(f"Error in inside middle third optimization for {name}: {e}")
                        error_area.error(f"Error in inside middle third optimization for {name}: {e}")

                    # Run outside middle third optimization
                    try:
                        results_outside = foundation_outside_middle_third_rect_df(group, q_max, Z_min, Z_max, X_min, X_max, Y_min, Y_max)
                        if not results_outside.empty:
                            max_X_row_outside = results_outside.loc[results_outside['X'].idxmax()]
                            result_outside = {
                                'X': round(max_X_row_outside['X'], 2),
                                'Y': round(max_X_row_outside['Y'], 2),
                                'Z': round(max_X_row_outside['Z'], 2),
                                'e_x': round(max_X_row_outside['e_x'], 2),
                                'e_y': round(max_X_row_outside['e_y'], 2)

                            }
                        else:
                            print(f"No feasible solution for outside middle third optimization for {name}.")
                            progress_area.info(f"No feasible solution for outside middle third optimization for {name}.")
                    except Exception as e:
                        print(f"Error in outside middle third optimization for {name}: {e}")
                        #progress_area.info(f"Error in outside middle third optimization for {name}: {e}")
                        error_area.error(f"Error in outside middle third optimization for {name}: {e}")

                    # Append the combined results
                    all_results.append({
                        'Name': name,
                        # Inside middle third results
                        'X_in': result_inside['X'],
                        'Y_in': result_inside['Y'],
                        'Z_in': result_inside['Z'],
                        'e_x_in': result_inside['e_x'],
                        'e_y_in': result_inside['e_y'],

                        # Outside middle third results
                        'X_out': result_outside['X'],
                        'Y_out': result_outside['Y'],
                        'Z_out': result_outside['Z'],
                        'e_x_out': result_outside['e_x'],
                        'e_y_out': result_outside['e_y']

                    })

                # Convert all results into a DataFrame
                final_results_df = pd.DataFrame(all_results)

                return final_results_df
            ###################################################################################################################
            ###################################################################################################################
            ### START 
            # Inputs
            #q_max=150 # Maximum Bearing Pressure

            #Z_min=0.75 # Min Z dimension
            #Z_max=150  # Max Z dimension

            #X_min=1 # Min X dimension
            #X_max=150 # Min X dimension

            #Y_min=1  # Min Y dimension
            #Y_max=150 # Min Y dimension

            #pQ=1.0 #Load factor

            #FoS=1.25 #Factor of Safety

            #u=0.45 # Friction Coefficient

            #Square = True #Use to run either square or rectangular optimisation
            ###################################################################################################################
            ###################################################################################################################
            ### Run Optimization
            with st.spinner("Optimising Please Wait..."):
                if Square:
                        #info_area.info("ℹ️ Square foundations optimisation running, please wait...")
                        #run= input("Running square foundations optimisation. Type yes to continue...").lower()
                        run="yes"
                        if run == "yes":
                            print("Optimisation running, please wait...")
                            
                            # Apply the combined optimization process
                            optimized_results_df = optimize_foundations_square(df_simplified, pQ, q_max, Z_min, Z_max, X_min, X_max)

                            # Display the final results
                            print(optimized_results_df)
                            st.subheader("Optimised Foundation Dimensions")
                            st.dataframe(optimized_results_df)
                            optimized_results_df.to_excel(optimised_output_path)
                        else:
                            print("Operation aborted!")
                            #sys.exit()
                        
                else:   
                        #run= input("Running rectangular foundations optimisation. Type yes to continue...").lower()
                        run="yes"
                        if run == "yes":
                            print("Optimisation running, please wait...")
                            # Apply the combined optimization process
                            optimized_results_df = optimize_foundations_rect(df_simplified, pQ, q_max, Z_min, Z_max, X_min, X_max, Y_min, Y_max)

                            # Display the final results
                            print(optimized_results_df)
                            st.subheader("Optimised Foundation Dimensions")
                            st.dataframe(optimized_results_df)
                            optimized_results_df.to_excel(optimised_output_path)
                        else:
                            print("Operation aborted!")
                            #sys.exit()
            ###################################################################################################################
            ###################################################################################################################
            # END of TOM's code
            info_area.success("✅ Optimisation Complete")
            st.session_state.optimisation_started = False
            if st.button("OK"):
                st.rerun()
                
