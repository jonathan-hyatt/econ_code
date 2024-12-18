cap log close
//Change the directory for the duration of the do file//
cd "C:\Users\hyattjon\Box\Fall 2023\Econ 398\Final Project"

//Start the log file//
log using "output_from_regression\Final Project.txt",replace

//Import Final Dataset//
use data\final_data.dta, clear

cd "C:\Users\hyattjon\Box\Fall 2023\Econ 398\Final Project\output_from_regression"
//Drop NULL Retention Rates//
drop if retention == "NULL"

//Make data the correct type//
local typelist retention gradrate loan_ever pell_ever faminc adm_rate cost_attend avgfacsal

foreach x in `typelist' {
    destring `x', replace force
}

//Organize Dataset so it's easier to read//
sort fips instnm year

//Replace any missing values with the mean of the variable
foreach var in adm_rate cost_attend loan_ever pell_ever faminc avgfacsal gradrate{
	egen mean = mean(`var')
	replace `var' = mean if missing(`var')
	drop mean
}

//Create variable for treatment and which year the state was treated//
gen treated = 0
replace treated = 2013 if fips == 8 | fips == 53
replace treated = 2015 if fips == 2 | fips == 11 | fips == 41
replace treated = 2017 if fips == 6 | fips == 23 | fips == 32
replace treated = 2019 if fips == 50 | fips == 26 | fips == 25
//Adjust years to show years before and after treatment//

//Create an indicator for being a close state//
generate close = 0
replace close = 1 if fips == 8 | fips == 2 | fips == 23 | fips == 25

gen years_before=.
*Loops through the treated states to show years pre and post treatment
foreach x in 2 6 8 53 11 41 23 32 50 26 25{
	replace years_before = 0 if year==treated & fips==`x' 
	replace years_before = year - treated if years_before!=0 & fips==`x'
}

replace years_before = year-2017 if years_before==.

//Print out graphs used to check for balance//
sum if treat !=0
sum if treat ==0
  
//Setting variables for XTDIDRegression////
ssc install event_plot
encode instnm, generate(inst)
xtset fips
global control_list adm_rate cost_attend avgfacsal urate gdp med_hs_income faminc ownership pell_ever

//Basic Regression//
*state level controls
xtdidregress (retention med_hs_income urate gdp totemploy) (legal), group(fips) time(year)
*state and institution controls
xtdidregress (retention $control_list) (legal), group(fips) time(year)
*create residuals variable
predict basic_resid, residuals

//Robust standard errors//
xtdidregress (retention $control_list) (legal), group(fips) time(year) vce(robust)
*create residuals variable
predict robust_resid, residuals

//Wild Bootstrap//
xtdidregress (retention $control_list) (legal), group(fips) time(year) vce(bootstrap)
*create residuals variable
predict bootstrap_resid, residuals

//Aggregate to try to eliminate collinearlity//
*Standard
xtdidregress (retention $control_list) (legal), group(fips) time(year) aggregate(standard)
*Dlang
xtdidregress (retention $control_list) (legal), group(fips) time(year) aggregate(dlang)

//Only the close states//
xtdidregress (retention $control_list) (legal) if close ==1 | treated == 0, group(fips) time(year) aggregate(standard)

*if we want to go back to the graduation rate we can change the gen to be the mean graduation rate and then run the same code
//Create averages for simpler graphing//
egen control = mean (robust_resid)  if treated == 0 , by (year)
egen all_treated = mean (robust_resid) if treated !=0, by (years_before)
egen treated_2013 = mean (robust_resid) if treated == 2013, by (year)
egen treated_2015 = mean (robust_resid) if treated == 2015, by (year)
egen treated_2017 = mean (robust_resid) if treated == 2017, by (year)
egen treated_2019 = mean (robust_resid) if treated == 2019, by (year)

xtset fips

global varlist adm_rate cost_attend avgfacsal urate gdp med_hs_income faminc ownership pell_ever retention

//Here is the hard coded part becuase i couldn't figure out the loop//
//Retention//
xtdidregress (retention $control_list) (legal), group(fips) time(year) vce(robust)
predict retention_robust_resid, residuals

//Create averages for simpler graphing//
egen retention_control = mean (robust_resid)  if treated == 0 , by (year)
egen retention_all_treated = mean (robust_resid) if treated !=0, by (year)
egen retention_treated_2013 = mean (robust_resid) if treated == 2013, by (year)
egen retention_treated_2015 = mean (robust_resid) if treated == 2015, by (year)
egen retention_treated_2017 = mean (robust_resid) if treated == 2017, by (year)
egen retention_treated_2019 = mean (robust_resid) if treated == 2019, by (year)

sort year
twoway (line retention_control years_before) ///
(line retention_treated_2013 years_before, lpattern(dash)) ///
(line retention_treated_2015 years_before, lpattern(longdash)) ///
(line retention_treated_2017 years_before, lpattern(dash_dot)) ///
(line retention_treated_2019 years_before), ///
xline(0) title("Retention by Institution") ///
ytitle("Retention") xtitle("Years")
graph export "retention_graph.jpg", replace

//Adm Rate//
xtdidregress (adm_rate cost_attend avgfacsal urate gdp med_hs_income faminc ownership pell_ever retention) (legal), group(fips) time(year) vce(robust)
predict adm_rate_robust_resid, residuals

//Create averages for simpler graphing//
egen adm_rate_control = mean (robust_resid)  if treated == 0 , by (year)
egen adm_rate_all_treated = mean (robust_resid) if treated !=0, by (year)
egen adm_rate_treated_2013 = mean (robust_resid) if treated == 2013, by (year)
egen adm_rate_treated_2015 = mean (robust_resid) if treated == 2015, by (year)
egen adm_rate_treated_2017 = mean (robust_resid) if treated == 2017, by (year)
egen adm_rate_treated_2019 = mean (robust_resid) if treated == 2019, by (year)

sort year
twoway (line adm_rate_control years_before) ///
(line adm_rate_treated_2013 years_before, lpattern(dash)) ///
(line adm_rate_treated_2015 years_before, lpattern(longdash)) ///
(line adm_rate_treated_2017 years_before, lpattern(dash_dot)) ///
(line adm_rate_treated_2019 years_before), ///
xline(0) title("Admittance Rate by Institution") ///
ytitle("Admin Rate") xtitle("Years")
graph export "adm_rate_graph.jpg", replace

//Cost Attend//
xtdidregress (cost_attend avgfacsal urate gdp med_hs_income faminc ownership pell_ever retention adm_rate) (legal), group(fips) time(year) vce(robust)
predict cost_attend_robust_resid, residuals

//Create averages for simpler graphing//
egen cost_attend_control = mean (cost_attend_robust_resid)  if treated == 0 , by (year)
egen cost_attend_all_treated = mean (cost_attend_robust_resid) if treated !=0, by (year)
egen cost_attend_treated_2013 = mean (cost_attend_robust_resid) if treated == 2013, by (year)
egen cost_attend_treated_2015 = mean (cost_attend_robust_resid) if treated == 2015, by (year)
egen cost_attend_treated_2017 = mean (cost_attend_robust_resid) if treated == 2017, by (year)
egen cost_attend_treated_2019 = mean (cost_attend_robust_resid) if treated == 2019, by (year)

sort year
twoway (line cost_attend_control years_before) ///
(line cost_attend_treated_2013 years_before, lpattern(dash)) ///
(line cost_attend_treated_2015 years_before, lpattern(longdash)) ///
(line cost_attend_treated_2017 years_before, lpattern(dash_dot)) ///
(line cost_attend_treated_2019 years_before), ///
xline(0) title("Cost Attend by Institution") ///
ytitle("Cost Attend") xtitle("Years")
graph export "cost_attend_graph.jpg", replace

//Avg Faculty Salary//
xtdidregress (avgfacsal urate gdp med_hs_income faminc ownership pell_ever retention adm_rate cost_attend) (legal), group(fips) time(year) vce(robust)
predict avgfacsal_robust_resid, residuals

//Create averages for simpler graphing//
egen avgfacsal_control = mean (avgfacsal_robust_resid)  if treated == 0 , by (year)
egen avgfacsal_all_treated = mean (avgfacsal_robust_resid) if treated !=0, by (year)
egen avgfacsal_treated_2013 = mean (avgfacsal_robust_resid) if treated == 2013, by (year)
egen avgfacsal_treated_2015 = mean (avgfacsal_robust_resid) if treated == 2015, by (year)
egen avgfacsal_treated_2017 = mean (avgfacsal_robust_resid) if treated == 2017, by (year)
egen avgfacsal_treated_2019 = mean (avgfacsal_robust_resid) if treated == 2019, by (year)

sort year
twoway (line avgfacsal_control years_before) ///
(line avgfacsal_treated_2013 years_before, lpattern(dash)) ///
(line avgfacsal_treated_2015 years_before, lpattern(longdash)) ///
(line avgfacsal_treated_2017 years_before, lpattern(dash_dot)) ///
(line avgfacsal_treated_2019 years_before), ///
xline(0) title("Average Faculty Salary by Institution") ///
ytitle("Average Faculty Salary") xtitle("Years")
graph export "avgfacsal_graph.jpg", replace

//Unempoyment Rate//
xtdidregress (urate gdp med_hs_income faminc ownership pell_ever retention adm_rate cost_attend avgfacsal) (legal), group(fips) time(year) vce(robust)
predict urate_robust_resid, residuals

//Create averages for simpler graphing//
egen urate_control = mean (urate_robust_resid)  if treated == 0 , by (year)
egen urate_all_treated = mean (urate_robust_resid) if treated !=0, by (year)
egen urate_treated_2013 = mean (urate_robust_resid) if treated == 2013, by (year)
egen urate_treated_2015 = mean (urate_robust_resid) if treated == 2015, by (year)
egen urate_treated_2017 = mean (urate_robust_resid) if treated == 2017, by (year)
egen urate_treated_2019 = mean (urate_robust_resid) if treated == 2019, by (year)

sort year
twoway (line urate_control years_before) ///
(line urate_treated_2013 years_before, lpattern(dash)) ///
(line urate_treated_2015 years_before, lpattern(longdash)) ///
(line urate_treated_2017 years_before, lpattern(dash_dot)) ///
(line urate_treated_2019 years_before), ///
xline(0) title("Unempoyment Rate by State") ///
ytitle("Unempoyment") xtitle("Years")
graph export "urate_graph.jpg", replace


//GDP//
xtdidregress (gdp med_hs_income faminc ownership pell_ever retention adm_rate cost_attend avgfacsal urate) (legal), group(fips) time(year) vce(robust)
predict gdp_robust_resid, residuals

//Create averages for simpler graphing//
egen gdp_control = mean (gdp_robust_resid)  if treated == 0 , by (year)
egen gdp_all_treated = mean (gdp_robust_resid) if treated !=0, by (year)
egen gdp_treated_2013 = mean (gdp_robust_resid) if treated == 2013, by (year)
egen gdp_treated_2015 = mean (gdp_robust_resid) if treated == 2015, by (year)
egen gdp_treated_2017 = mean (gdp_robust_resid) if treated == 2017, by (year)
egen gdp_treated_2019 = mean (gdp_robust_resid) if treated == 2019, by (year)

sort year
twoway (line gdp_control years_before) ///
(line gdp_treated_2013 years_before, lpattern(dash)) ///
(line gdp_treated_2015 years_before, lpattern(longdash)) ///
(line gdp_treated_2017 years_before, lpattern(dash_dot)) ///
(line gdp_treated_2019 years_before), ///
xline(0) title("GDP by State") ///
ytitle("GDP") xtitle("Years")
graph export "gdp_graph.jpg", replace

//Median Household Income//
xtdidregress (med_hs_income faminc ownership pell_ever retention adm_rate cost_attend avgfacsal urate gdp) (legal), group(fips) time(year) vce(robust)
predict med_hs_income_robust_resid, residuals

//Create averages for simpler graphing//
egen med_hs_income_control = mean (med_hs_income_robust_resid)  if treated == 0 , by (year)
egen med_hs_income_all_treated = mean (med_hs_income_robust_resid) if treated !=0, by (year)
egen med_hs_income_treated_2013 = mean (med_hs_income_robust_resid) if treated == 2013, by (year)
egen med_hs_income_treated_2015 = mean (med_hs_income_robust_resid) if treated == 2015, by (year)
egen med_hs_income_treated_2017 = mean (med_hs_income_robust_resid) if treated == 2017, by (year)
egen med_hs_income_treated_2019 = mean (med_hs_income_robust_resid) if treated == 2019, by (year)

sort year
twoway (line med_hs_income_control years_before) ///
(line med_hs_income_treated_2013 years_before, lpattern(dash)) ///
(line med_hs_income_treated_2015 years_before, lpattern(longdash)) ///
(line med_hs_income_treated_2017 years_before, lpattern(dash_dot)) ///
(line med_hs_income_treated_2019 years_before), ///
xline(0) title("Median Household Income by State") ///
ytitle("Median Household Income") xtitle("Years")
graph export "med_hs_income_graph.jpg", replace

//Faminc//
xtdidregress (faminc ownership pell_ever retention adm_rate cost_attend avgfacsal urate gdp med_hs_income) (legal), group(fips) time(year) vce(robust)
predict faminc_robust_resid, residuals

//Create averages for simpler graphing//
egen faminc_control = mean (faminc_robust_resid)  if treated == 0 , by (year)
egen faminc_all_treated = mean (faminc_robust_resid) if treated !=0, by (year)
egen faminc_treated_2013 = mean (faminc_robust_resid) if treated == 2013, by (year)
egen faminc_treated_2015 = mean (faminc_robust_resid) if treated == 2015, by (year)
egen faminc_treated_2017 = mean (faminc_robust_resid) if treated == 2017, by (year)
egen faminc_treated_2019 = mean (faminc_robust_resid) if treated == 2019, by (year)

sort year
twoway (line faminc_control years_before) ///
(line faminc_treated_2013 years_before, lpattern(dash)) ///
(line faminc_treated_2015 years_before, lpattern(longdash)) ///
(line faminc_treated_2017 years_before, lpattern(dash_dot)) ///
(line faminc_treated_2019 years_before), ///
xline(0) title("Family Income by Institution") ///
ytitle("Family Income") xtitle("Years")
graph export "faminc_graph.jpg", replace

//ownership//
xtdidregress (ownership pell_ever retention adm_rate cost_attend avgfacsal urate gdp med_hs_income faminc) (legal), group(fips) time(year) vce(robust)
predict ownership_robust_resid, residuals

//Create averages for simpler graphing//
egen ownership_control = mean (ownership_robust_resid)  if treated == 0 , by (year)
egen ownership_all_treated = mean (ownership_robust_resid) if treated !=0, by (year)
egen ownership_treated_2013 = mean (ownership_robust_resid) if treated == 2013, by (year)
egen ownership_treated_2015 = mean (ownership_robust_resid) if treated == 2015, by (year)
egen ownership_treated_2017 = mean (ownership_robust_resid) if treated == 2017, by (year)
egen ownership_treated_2019 = mean (ownership_robust_resid) if treated == 2019, by (year)

sort year
twoway (line ownership_control years_before) ///
(line ownership_treated_2013 years_before, lpattern(dash)) ///
(line ownership_treated_2015 years_before, lpattern(longdash)) ///
(line ownership_treated_2017 years_before, lpattern(dash_dot)) ///
(line ownership_treated_2019 years_before), ///
xline(0) title("Ownership Type by Institution") ///
ytitle("Ownership") xtitle("Years")
graph export "ownership_graph.jpg", replace

//pell_ever//
xtdidregress (pell_ever retention adm_rate cost_attend avgfacsal urate gdp med_hs_income faminc ownership) (legal), group(fips) time(year) vce(robust)
predict pell_ever_robust_resid, residuals

//Create averages for simpler graphing//
egen pell_ever_control = mean (pell_ever_robust_resid)  if treated == 0 , by (year)
egen pell_ever_all_treated = mean (pell_ever_robust_resid) if treated !=0, by (year)
egen pell_ever_treated_2013 = mean (pell_ever_robust_resid) if treated == 2013, by (year)
egen pell_ever_treated_2015 = mean (pell_ever_robust_resid) if treated == 2015, by (year)
egen pell_ever_treated_2017 = mean (pell_ever_robust_resid) if treated == 2017, by (year)
egen pell_ever_treated_2019 = mean (pell_ever_robust_resid) if treated == 2019, by (year)

sort year
twoway (line pell_ever_control years_before) ///
(line pell_ever_treated_2013 years_before, lpattern(dash)) ///
(line pell_ever_treated_2015 years_before, lpattern(longdash)) ///
(line pell_ever_treated_2017 years_before, lpattern(dash_dot)) ///
(line pell_ever_treated_2019 years_before), ///
xline(0) title("Pell Grant Ever by Institution") ///
ytitle("Pell Grant Ever") xtitle("Years")
graph export "pell_ever_graph.jpg", replace


//Sort so the graph looks clean//
sort years_before
//All treated states//
twoway (line control years_before) (line all_treated years_before) , xline(0) yscale(range(-0.1,0.1)) ylabel(-0.1(0.02)0.1) title("Retention Rates by State") ytitle("Retention Rates") xtitle("Years")
graph export "all_treated_graph.jpg", replace

*the graph still has a lot of noise even when we just graph the residuals
//Create graph centered on treatement time//
twoway (line control years_before) (line all_treated years_before) (line treated_2013 years_before) (line treated_2015 years_before) (line treated_2017 years_before) (line treated_2019 years_before), xline(0) title("Retention Rates by State") ytitle("Retention Rates") xtitle("Years")
graph export "parallel_trends_centered_graph.jpg", replace

//Create graph with lines at treatment times and labels//
twoway (line control year) (line treated_2013 year) (line treated_2015 year) (line treated_2017 year) (line treated_2019 year), xline(2013) xline(2015) xline(2017) xline(2019) title("Retention Rates by State") ytitle("Retention Rates") xtitle("Years")
graph export "parallel_trends_basic_graph.jpg", replace

//Specific treated groups//
//2013//
twoway (line control years_before) (line treated_2013 years_before, lpattern(dash)) , xline(0) title("Retention Rates by State") ytitle("Retention Rates") xtitle("Years")
graph export "2013_centered_graph.jpg", replace
twoway (line control year) (line treated_2013 year, lpattern(dash)) , xline(2013) title("Retention Rates by State") ytitle("Retention Rates") xtitle("Years")
graph export "2013_basic_graph.jpg", replace
//2015//
twoway (line control years_before) (line treated_2015 years_before, lpattern(dash)) , xline(0) title("Retention Rates by State") ytitle("Retention Rates") xtitle("Years")
graph export "2015_centered_graph.jpg", replace
twoway (line control year) (line treated_2015 year, lpattern(dash)) , xline(2015) title("Retention Rates by State") ytitle("Retention Rates") xtitle("Years")
graph export "2015_basic_graph.jpg", replace
//2017//
twoway (line control years_before) (line treated_2017 years_before, lpattern(dash)) , xline(0) title("Retention Rates by State") ytitle("Retention Rates") xtitle("Years")
graph export "2017_centered_graph.jpg", replace
twoway (line control year) (line treated_2017 year, lpattern(dash)) , xline(2017) title("Retention Rates by State") ytitle("Retention Rates") xtitle("Years")
graph export "2017_basic_graph.jpg", replace
//2019//
twoway (line control years_before) (line treated_2019 years_before, lpattern(dash)) , xline(0) title("Retention Rates by State") ytitle("Retention Rates") xtitle("Years")
graph export "2019_centered_graph.jpg", replace
twoway (line control year) (line treated_2019 year, lpattern(dash)) , xline(2019) title("Retention Rates by State") ytitle("Retention Rates") xtitle("Years")
graph export "2019_basic_graph.jpg", replace

//FOR REFERENCE : Jane's regression//
*xi: regress retention legal i.inst i.year

//Close the log//
log close