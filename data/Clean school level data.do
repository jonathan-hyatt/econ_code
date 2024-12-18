
// Directory 
cd "C:\Users\gradenpj\Box\Final Project\data"

// Prepare for loop
clear all 
set more off

local years "2000_01 2001_02 2002_03 2003_04 2004_05 2005_06 2006_07 2007_08 2008_09 2009_10 2010_11 2011_12 2012_13 2013_14 2014_15 2015_16 2016_17 2017_18 2018_19 2019_20 2020_21 2021_22"

foreach year in `years' {
	//Import csv files 2000-2022
	import delimited "MERGED`year'_PP.csv", clear 
	
	// Select variables 
	keep instnm stabbr main preddeg highdeg control st_fips ccugprof ccsizset adm_rate satvrmid satmtmid satwrmid sat_avg costt4_a costt4_p avgfacsal ret_ft4 ret_pt4 c150_4 c150_4_white c150_4_black c150_4_hisp c150_4_asian c150_4_aian c150_4_nhpi lo_inc_comp_orig_yr6_rt md_inc_comp_orig_yr6_rt hi_inc_comp_orig_yr6_rt female_comp_orig_yr6_rt male_comp_orig_yr6_rt loan_ever pell_ever faminc md_faminc median_hh_inc unemp_rate stufacr control_peps
	
	// Filter for main campuses only
	drop if main == 0
	
	// Drop universities that offer associates degrees or less as their highest degrees
	drop if highdeg <3
	
	// Rename variable that indicates if private or public
	rename control ownership
	
	// Drop Universities with 100% graduation rate? or null values?
	//drop if c150_4 == "1"
	
	//Generate year variable 
	gen year = "`year'"
	
	//Append to the main dataset 
	save year_`year', replace
 	
	
}

// Edit for format (destring certain variables in certain years)
clear 
use "year_2002_03.dta"
destring ownership, replace
save year_2002_03, replace

clear 
use "year_2006_07.dta"
destring ownership, replace
save year_2006_07, replace

clear 
use "year_2008_09.dta"
destring ownership, replace
save year_2008_09, replace

clear 
use "year_2011_12.dta"
destring st_fips, replace
save year_2011_12, replace


// Append all 
clear all 
set more off

local years "2000_01 2001_02 2002_03 2003_04 2004_05 2005_06 2006_07 2007_08 2008_09 2009_10 2010_11 2011_12 2012_13 2013_14 2014_15 2015_16 2016_17 2017_18 2018_19 2019_20 2020_21 2021_22"

foreach year in `years' {
	append using "year_`year'.dta", force 
}

// Adjust year 
gen yr = substr(year, 1, 4)
destring yr, replace
drop year 
rename yr year, replace

// Rename fips 
rename st_fips fips, replace 

// Four year only 
drop if highdeg == 4 


// Save
save main_dataset, replace 

//////////////////////////////

// Add Legalization Indicators 
// Marijuana Indicator (All Legalized in November)//

gen legal = ((year == 2013| year == 2014| year == 2015| year == 2016| year == 2017| year == 2018| year == 2019| year == 2020| year == 2021) & (fips == 8 | fips == 53)) //Colorado, Washington

replace legal = 1 if (year == 2015| year == 2016| year == 2017| year == 2018| year == 2019| year == 2020| year == 2021) & (fips == 2| fips == 11| fips == 41) //Alaska, DC, Oregon 

replace legal = 1 if (year == 2017| year == 2018| year == 2019| year == 2020| year == 2021) & (fips == 6| fips == 23| fips == 32) //California, Maine, Nevada

replace legal = 1 if (year == 2019| year == 2020| year == 2021) & (fips == 50| fips == 26| fips == 25) //Vernmont (Jan), Michigan, Massachusetts (August) 

replace legal = 1 if (year == 2020| year == 2021) & (fips == 17) // Illinois (May) 

replace legal = 1 if (year == 2021) & (fips == 34| fips == 30| fips == 4) //New Jersey, Montana, Arizona 

save main_dataset_cov.dta, replace


// Economic Covariates//



// import GDP// 
import delimited "(gdp) by state (dollars).csv", varnames(1) clear 
// reshape//
drop if fips == "Years"
destring fips, replace
rename fips year
drop if year == .
reshape long s, i(year) j(number)
rename number fips 
rename s gdp
destring gdp, replace
drop if year > 2021
drop if year < 2000

// Merge with main_dataset_cov//
merge m:m fips year using main_dataset_cov.dta
drop _merge
drop if fips > 55
drop if missing(legal)
save main_dataset_cov.dta, replace
 
// import unemployment rate and total employment//
import excel "Unemployment_rate.xlsx", sheet("ststdsadata") cellrange(A4:K30377) firstrow clear 
// clean//
rename Stateandarea state
rename Period year
rename D month
rename H totemploy
rename K urate
rename FIPSCode fips
keep fips state year totemploy urate
drop if fips == ""
destring fips year totemploy urate, replace
drop if state == "Los Angeles County"
drop if fips == 51000
drop if year < 2000
collapse (mean) totemploy urate, by ( fips state year)
drop state
drop if year > 2021

// merge with main_dataset_cov//
merge m:m fips year using main_dataset_cov.dta
drop _merge 
drop if missing(legal)
save main_dataset_cov.dta, replace
 
// import Median Household Income// 
import excel "Median_State_Income.xlsx", sheet("h08") cellrange(A8:Y60) firstrow clear 
// Reshape//
drop if State == "United States"
rename State state
rename Fips fips
reshape long yr, i(state) j(year)
rename yr med_hs_income
drop if year > 2021
drop state 

// Merge with main_dataset_cov //
merge m:m fips year using main_dataset_cov.dta
drop _merge 
drop if missing(legal)
save main_dataset_cov.dta, replace

// Rename key variables
rename ret_ft4 retention 
rename c150_4 gradrate
save main_dataset_cov.dta, replace


*************************************************************************

// If using retention rate
keep if year > 2003 

// How many NULL values in retention?
tab retention // 1,559/10,515 = 14.8% missing 

drop if retention != "NULL"

// What type of schools dropped?
tab instnm if retention == "NULL" 
export excel using "dropped", firstrow(variables) replace
tab ownership if retention == "NULL"
tab stabbr

replace adm_rate = "." if adm_rate=="NULL"
destring adm_rate, replace

replace avgfacsal = "." if avgfacsal=="NULL"
destring avgfacsal, replace

replace faminc = "." if (faminc=="NULL" | faminc == "PrivacySuppressed")
destring faminc, replace

replace costt4_a = "." if (costt4_a=="NULL" | costt4_a == "PrivacySuppressed")
destring costt4_a, replace

// Summarize 
sum adm_rate
sum avgfacsal
sum faminc
sum costt4_a

// Final Clean
clear 
use "C:\Users\gradenpj\Box\Final Project\data\main_dataset_cov.dta"

// Drop missing retention 
drop if retention == "NULL"

// Final Variables 

keep year fips med_hs_income totemploy urate gdp instnm stabbr preddeg ownership adm_rate avgfacsal costt4_a gradrate retention loan_ever pell_ever faminc legal 

// tab ownership
// tab stabbr
//
// replace adm_rate = "." if adm_rate=="NULL"
// destring adm_rate, replace
//
// replace avgfacsal = "." if avgfacsal=="NULL"
// destring avgfacsal, replace
//
// replace faminc = "." if (faminc=="NULL" | faminc == "PrivacySuppressed")
// destring faminc, replace
//
// replace costt4_a = "." if (costt4_a=="NULL" | costt4_a == "PrivacySuppressed")
// destring costt4_a, replace
//
//
// sum adm_rate
// sum avgfacsal
// sum faminc
// sum costt4_a

// Rename
rename stabbr state
rename costt4_a cost_attend
rename preddeg common_degree

// Save Data 

save final_data, replace
export excel using "final", firstrow(variables) replace



// End 

