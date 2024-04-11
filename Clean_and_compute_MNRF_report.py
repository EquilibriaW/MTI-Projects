# This is a sample Python script.
import copy

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.

import numpy as np

import pandas as pd

from numpy import dot
from numpy.linalg import norm




#Regionalize Data
#Project economic impacts
#Estimate provincial gov tax revenues
#Estimate three year average impacts per 1000^3m of timber harvest


#gdp in million dollar units


def compute_nominal_gdps(nominal_gdp_one_year, real_gdp, price_indices):
    ref_deflator = nominal_gdp_one_year/real_gdp[0] * 100
    deflators = price_indices[1:6] * (ref_deflator/price_indices[0])
    nominal_gdp = np.multiply(deflators, real_gdp)
    nominal_gdps_percentage_change = (nominal_gdp/nominal_gdp_one_year) - 1
    return nominal_gdps_percentage_change


def compute_provincial_ref_year_PIT(pit, primary_household_income, labour_income_impacts,
                                    provincial_PIT, total_PIT):
    pit_share = labour_income_impacts*pit/primary_household_income
    return pit_share * (provincial_PIT/total_PIT)

def compute_provincial_ref_year_CIT(taxable_income, GOS, provincial_CIT, total_CIT):
    cit_share = taxable_income/GOS
    return cit_share * (provincial_CIT/total_CIT)

def compute_projected_years(ref_year_figures, nominal_gdps_percentage_change):
    projected_years_figures = np.array([])
    for i in nominal_gdps_percentage_change:
        np.append(projected_years_figures, ref_year_figures*(i + 1))
    return projected_years_figures

def compute_harvest_normalized_table(IO_tables, total_dollar_outputs, total_forest_harvests):
    normalized_table = np.array([])
    for i in range(len(IO_tables)):
        normalized_table = IO_tables[i,:]*total_dollar_outputs[i]/total_forest_harvests[i]
    return normalized_table/3

def regionalize_data(regional_ratios, total_figures):
    regionalized_figures = np.array([])
    for ratio in regional_ratios:
        np.append(regionalized_figures, total_figures*ratio)
    return regionalized_figures



def generate_regional_ratios(path):
    cleaned_census_data = pd.read_csv(path).iloc[:, 0:15]
    cleaned_census_data.iloc[:,8:15] = cleaned_census_data.iloc[:,8:15].applymap(set_numeric)
    subtables = [cleaned_census_data.iloc[i:i + 4] for i in range(0, len(cleaned_census_data), 4)]
    results = []
    Northeast_salary = 0
    Northwest_salary = 0
    The_rest_salary = 0
    for subtable in subtables:
        value = 0
        for col in range(8,15):
            value += subtable.iloc[0, col] * subtable.iloc[1, col] + subtable.iloc[2, col] * subtable.iloc[3, col]
        if check_northwest(subtable.iloc[1, 1]):
            region = "Northwest"
            Northwest_salary += value
        elif check_northeast(subtable.iloc[1, 1]):
            region = "Northeast"
            Northeast_salary += value
        else:
            region = "The_Rest"
            The_rest_salary += value
        results.append((value, region))
    total = Northeast_salary + Northwest_salary + The_rest_salary
    return (results, [Northeast_salary/total, Northwest_salary/total, The_rest_salary/total])
    #returns a tuple. First is raw results, the second a list of ratios.

def set_numeric(entry):
    try:
        float(entry)
        return float(entry)
    except ValueError:
        return 0

def check_northwest(entry):
    elements_to_check = ["Thunder Bay", "Rainy River", "Kenora"]
    return any(element in str(entry) for element in elements_to_check)


def check_northeast(entry):
    elements_to_check = ["Nipissing", "Parry Sound", "Manitoulin", "Sudbury",
     "Greater Sudbury/ Grand Sudbury", "Timiskaming", "Cochrane" , "Algoma"]
    return any(element in str(entry) for element in elements_to_check)

def compute_harvest_volume_by_year(region):
    yearly_harvest_volumes = []
    for i in range(3):
        total = 0
        for FMU in region:
            total +=  FMU[1][i]
        yearly_harvest_volumes.append(total)
    return yearly_harvest_volumes

#This simply assigns every census division to one of three categories: Northeast, northewest,
# and the rest of the eight economic regions as the third category. Then we simply sum
# over all relevant naics industries (correspondent with  an

#want to return an array of regional ratios.

def compute_price_index(raw_materials_price_index , CPI_services, industrial_price_index):
    return (raw_materials_price_index + CPI_services + industrial_price_index)/3

def make_IOIC_coefficients_tensor(path):
    read_IOIC_multipliers = pd.read_csv(path).iloc[4, 2:]
    read_IOIC_multipliers = np.char.replace(read_IOIC_multipliers.to_numpy().astype(str), ",", "").astype(float)
    subarrays = [read_IOIC_multipliers[i:i + 4] for i in range(0, len(read_IOIC_multipliers), 4)]
    matrices = [subarrays[i:i + 7] for i in range(0, len(subarrays), 7)]
    matrices = [m for m in matrices if len(m) == 7 and all(len(sub) == 4 for sub in m)]
    tensors = [matrices[i:i + 5] for i in range(0, len(matrices), 5)]
    tensors = [t for t in tensors if len(t) == 5]
    tensor_4d = np.array(tensors)
    return tensor_4d
    # On the impact axis, it is direct, indirect, induced; in that order.
    # Order of IO impact is BS113000, BS115300, BS321100, BS321200, BS321900,
    # BS322100, BS322200
    # On variable axis, it is GDP, Taxes on Products, Labour income,
    # Taxes on Production, Jobs.
    # On year axis, it has 4 years: 2017-2020.
    # 4 by 7 by 5 by 3.. in that order.


#We must match "regionalized

def clean_census_data(path):
    census_divisions_table = pd.read_csv("/Users/alex/Downloads/2021StatCan.csv", encoding_errors='replace')
    condition1 = census_divisions_table[census_divisions_table.columns[0]].str.contains(
    '|'.join([
        "Algoma",
        "Brant",
        "Bruce",
        "Chatham-Kent",
        "Cochrane",
        "Dufferin",
        "Durham",
        "Elgin",
        "Essex",
        "Frontenac",
        "Greater Sudbury",
        "Grey",
        "Haldimand-Norfolk",
        "Haliburton",
        "Halton",
        "Hamilton",
        "Hastings",
        "Huron",
        "Kawartha Lakes",
        "Kenora",
        "Lambton",
        "Lanark",
        "Leeds and Grenville",
        "Lennox and Addington",
        "Manitoulin",
        "Middlesex",
        "Muskoka",
        "Niagara",
        "Nipissing",
        "Northumberland",
        "Ottawa",
        "Oxford",
        "Parry Sound",
        "Peel",
        "Perth",
        "Peterborough",
        "Prescott and Russell",
        "Prince Edward",
        "Rainy River",
        "Renfrew",
        "Simcoe",
        "Stormont, Dundas and Glengarry",
        "Sudbury",
        "Thunder Bay",
        "Timiskaming",
        "Toronto",
        "Waterloo",
        "Wellington",
        "York"
    ]), na=False)

    condition2 = census_divisions_table[census_divisions_table.columns[2]].str.contains(
        "Worked part year and/or part time|Worked full year full time", na=False)

    condition3 = census_divisions_table[census_divisions_table.columns[1]].str.contains(
        "Total Labour Force Status", na=False)

    condition4 = ~census_divisions_table[census_divisions_table.columns[0]].str.contains("Unorganized", na=False)

    combined_condition = condition1 & condition2 & condition3 & condition4

    census_divisions_table = census_divisions_table[combined_condition]

    #Also a bit I did manually. End up with only 48/49 census divisions; exclude
    #sudbury because too small.


def compute_harvest_volumes_byregion_byyear(volumes_path, regioncode_path):
    #Harvest Volume Calculation uses 2020 - 2022 data.
    harvest_volumes = pd.read_csv(volumes_path).iloc[1361:4289,:]
    FMU_region_codes = pd.read_csv(regioncode_path)
    FMU_region_codes = FMU_region_codes.groupby('LEAD_MNR_REGION_NAME')
    regions = [group for _, group in FMU_region_codes]
    region_codes = []
    for region in regions:
        region_codes.append(region["FMU_CODE"].tolist())
    # First index is Northeast, second is Northwest, third is southern.
    harvest_volumes["Volume"] = harvest_volumes["Volume"].astype(float)
    FMU_list = []
    grouping = harvest_volumes.groupby('MUNO')
    subtables = [group for _, group in grouping]
    for subtable in subtables:
        volume_over_years = subtable.groupby('Year')["Volume"].sum()
        volume_over_years = volume_over_years.tolist()
        FMU = (subtable.iloc[1, 2], volume_over_years)
        FMU_list.append(FMU)
    Northeast = []
    Northwest = []
    Southern = []
    for FMU in FMU_list:
        if FMU[0] in region_codes[0]:
            Northeast.append(FMU)
        elif FMU[0] in region_codes[1]:
            Northwest.append(FMU)
        else:
            Southern.append(FMU)
    FMU_994 = Northwest[len(Northwest)-1]
    Northwest = Northwest[1:len(Northwest) - 1]
    region_yearly_totals = [compute_harvest_volume_by_year(Northeast),
                            compute_harvest_volume_by_year(Northwest),
                            compute_harvest_volume_by_year(Southern)]
    region_yearly_totals[1][1] = FMU_994[1][0] + region_yearly_totals[1][1]
    region_yearly_totals[1][1] = FMU_994[1][1] + region_yearly_totals[1][1]
    #Essentially, FMU_994 doesn't exist in year 2020. So this just incorporates its
    #info into the other two years.
    return region_yearly_totals

def compute_2017_2020_five_variable_impact(path, IOIC_coefficient_table):
    seven_industry_basic_gdp_2017_to_2022 = pd.read_csv(
        path).iloc[4: , 2:]
    seven_industry_basic_gdp_2017_to_2020 = seven_industry_basic_gdp_2017_to_2022.iloc[:4]
    seven_industry_basic_gdp_2017_to_2020 = np.char.replace(
        seven_industry_basic_gdp_2017_to_2020.to_numpy().astype(str), ",", "").astype(float)
    #seven_industry_basic_gdp_2021_to_2022 = seven_industry_basic_gdp_2017_to_2022.iloc[5:]
    #seven_industry_basic_gdp_2021_to_2022 = np.char.replace(
        #seven_industry_basic_gdp_2021_to_2022.to_numpy().astype(str), ",", "").astype(float)
    #GDP in millions of dollars.
    computed_final_impacts_2017_to_2020 = np.array([])
    for k in range(4):
        year_data = np.array([])
        for i in range(3):
            impact_type_data = np.array([])
            for j in range(5):
                impact_variable = np.multiply(seven_industry_basic_gdp_2017_to_2020[k], IOIC_coefficient_table[i, j].T[k])
                if impact_type_data.size == 0:
                    impact_type_data = impact_variable[np.newaxis, :]
                else:
                    impact_type_data = np.concatenate((impact_type_data, impact_variable[np.newaxis, :]), axis=0)
            if year_data.size == 0:
                year_data = impact_type_data[np.newaxis, :]
            else:
                year_data = np.concatenate((year_data, impact_type_data[np.newaxis, :]), axis=0)
        if computed_final_impacts_2017_to_2020.size == 0:
            computed_final_impacts_2017_to_2020 = year_data[np.newaxis, :]
        else:
            computed_final_impacts_2017_to_2020 = np.concatenate((computed_final_impacts_2017_to_2020, year_data[np.newaxis, :]), axis=0)
    return computed_final_impacts_2017_to_2020

def compute_deflator(real_gdp_path, raw_materials_price_index_path, CPI_services_path, industrial_price_index_path):

    real_gdp = pd.read_csv(real_gdp_path).iloc[6, 1:]
    real_gdp = real_gdp.to_numpy().astype(str)
    real_gdp = np.char.replace(real_gdp, ",", "")
    real_gdp = real_gdp.astype(float)
    nominal_gdp_one_year = 764464.8
    raw_materials_price_index = pd.read_csv(raw_materials_price_index_path).iloc[-2, ::12].to_numpy()
    CPI_services = pd.read_csv(CPI_services_path).iloc[-1].to_numpy()
    industrial_price_index = pd.read_csv(industrial_price_index_path).iloc[9, ::12].to_numpy()
    raw_materials_price_index = raw_materials_price_index[1:].astype(float)
    CPI_services = CPI_services[1:].astype(float)
    industrial_price_index = industrial_price_index[1:].astype(float)
    price_index = compute_price_index(raw_materials_price_index, CPI_services, industrial_price_index)
    nominal_gdps = compute_nominal_gdps(nominal_gdp_one_year, real_gdp, price_index)
    return  nominal_gdps/100
    #2018 - 2022






if __name__ == '__main__':
    IOIC_coefficient_table = make_IOIC_coefficients_tensor("/Users/alex/Downloads/IOIC_impacts.csv")
    regional_ratios = generate_regional_ratios("/Users/alex/Downloads/census_data.csv")[1]
    harvest_numbers = compute_harvest_volumes_byregion_byyear("/Users/alex/Downloads/FMU_Harvest_Volumes.csv",
                                                              "/Users/alex/Downloads/FMU_regioncodes.csv")
    five_variable_impacts = compute_2017_2020_five_variable_impact("/Users/alex/Downloads/seven_industries_output_2017_2020.csv",
                                                                   IOIC_coefficient_table)
    deflator = compute_deflator("/Users/alex/Downloads/3610040201-eng.csv", "/Users/alex/Downloads/Raw_materials.csv",
                                "/Users/alex/Downloads/CPI_services.csv", "/Users/alex/Downloads/Industrial_price_index.csv")
    update_deflator =  np.concatenate(([1.0], deflator))


    # On the impact axis, it is direct, indirect, induced; in that order.
    # Order of IO impact is BS113000, BS115300, BS321100, BS321200, BS321900,
    # BS322100, BS322200
    # On variable axis, it is GDP, Taxes on Products, Labour income,
    # Taxes on Production, Jobs.
    # On year axis, it has 4 years: 2017-2020.
    # year, impact axis, variable type, impact on industries in
    # that order by indexing.

    #variable type, impact axis, impact on industries, year, region


    update_1 = five_variable_impacts*update_deflator[:4].reshape(4,1,1,1)
    update_2 = (five_variable_impacts[0,:,:,:]*update_deflator[4])[np.newaxis, ...]
    update_3 = (five_variable_impacts[0,:,:,:]*update_deflator[5])[np.newaxis, ...]


    impacts_2017_2022 = np.concatenate((update_1, update_2, update_3), axis =0)

    impacts_2017_2022_regionalized = np.concatenate(((impacts_2017_2022*regional_ratios[0])[..., np.newaxis], (impacts_2017_2022*regional_ratios[1])[..., np.newaxis],
                                                     (impacts_2017_2022*regional_ratios[2])[..., np.newaxis]), axis =4)

    impacts_2017_2022_regionalized = np.transpose(impacts_2017_2022_regionalized, (1, 2, 3, 0, 4))

    impacts_2020_2022_regionalized_harvest = copy.deepcopy(impacts_2017_2022_regionalized[:,:,:,3:5,:])




    for i in range(2):
        for j in range(3):
            impacts_2020_2022_regionalized_harvest[:, :, :, i, j] = \
                impacts_2020_2022_regionalized_harvest[:, :, :, i, j]/harvest_numbers[i][j]*1000

    impacts_2020_2022_regionalized_harvest[:, 0:4, :, :, :] = impacts_2020_2022_regionalized_harvest[:, 0:4, :, :, :]*1000

    impacts_2020_2022_regionalized_harvest = impacts_2020_2022_regionalized_harvest.transpose(1, 0, 2, 3, 4)

    print(impacts_2020_2022_regionalized_harvest.shape)

    impact_type_labels = ['Direct', 'Indirect', 'Induced']
    variable_labels = ['GDP', 'Taxes_on_Products', 'Labour_income', 'Taxes_on_Production', 'Jobs']
    industry_impact_labels = ['BS113000', 'BS115300', 'BS321100', 'BS321200', 'BS321900','BS322100', 'BS322200']
    year_labels_full = ['2017', '2018', '2019', '2020', '2021', '2022']
    year_labels_half = ['2020', '2021']
    region_labels = ['Northeast', 'Northwest', 'Southern']

    row_index_1 = pd.MultiIndex.from_product([impact_type_labels, industry_impact_labels, year_labels_half, region_labels ],
                                             names=['impact_type', 'industry_impact', 'year', 'region'])

    column_index_1 = pd.Index(variable_labels, name='variable')


    row_index = pd.MultiIndex.from_product([impact_type_labels, variable_labels, industry_impact_labels,year_labels_full, region_labels ],
                                           names=['impact_type', 'variable', 'industry_impact', 'year', 'region'])

    column_index = ["column_index"]


    df = pd.DataFrame(impacts_2017_2022_regionalized.reshape(-1, 1),
                      index=row_index, columns=column_index)

    df_1 = pd.DataFrame(impacts_2020_2022_regionalized_harvest.reshape(-1, impacts_2020_2022_regionalized_harvest.shape[0]),
                        index=row_index_1, columns=column_index_1)


    df_1.to_csv('/Users/alex/Downloads/harvest.csv')

    df.to_csv('/Users/alex/Downloads/non_harvest.csv')































































