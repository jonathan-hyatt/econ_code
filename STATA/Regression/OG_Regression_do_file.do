cap log close
//Change the directory for the duration of the do file//
//Use this when on Physical Computer//
*cd "C:\Users\hyattjon.BYU\Box\Econ 398\Final Project"
//Use when on Cloud Drive//
cd "C:\Users\hyattjon\Box\Econ 398\Final Project"

//Start the log file//
log using "Final Project.smcl",replace

cd "C:\Users\hyattjon\Box\Econ 398\Final Project\data"
//Import Final Dataset//
use final_data.dta, clear

//Drop NULL Value//
drop if retention == "NULL"
//Make all data the right type//
destring retention, replace
destring gradrate, replace force
destring loan_ever, replace force
destring pell_ever, replace force
destring faminc, replace force
destring adm_rate, replace force
destring cost_attend, replace force
destring avgfacsal, replace force


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
*maybe we add 2019 depending on what Dr. Pope says

gen years_before=.

foreach x in 2 6 8 50 53 11 41 23 32 26 25{
	replace years_before = 0 if year==treated & fips==`x' 
	replace years_before = year - treated if years_before!=0 & fips==`x'
}

replace years_before = year-2016 if years_before==.


//Print out graphs used to check for balance//
sum if treat !=0
sum if treat ==0

//Setting variables for XTDIDRegression////
ssc install event_plot
encode instnm, generate(inst)
xtset fips

//Basic Regression//
xtdidregress (retention adm_rate cost_attend avgfacsal urate gdp med_hs_income faminc ownership pell_ever) (legal), group(fips) time(year)

*only use pell and not loan 
*add code that shows
*use the same covariates for all regressions standardize the covariates 
*adm_rate, cost_attend, avgfacsal, urate, gdp, med_hs_income, faminc, ownership, pell_ever
*one that state effects  one that is instutions effects as well
*	check if the covariates that we are imputing are effective in changing the regression
*	check for each imputed variables Check to see if there is a difference between the imputed and non imputed regression

//FOR REFERENCE : Jane's regression//
*xi: regress retention legal i.inst i.year

*the only variables I could not use were the loan_ever pell_ever and faminc I got a syntax error
//Robust standard errors//
xtdidregress (retention adm_rate cost_attend avgfacsal urate gdp med_hs_income faminc ownership pell_ever) (legal), group(fips) time(year) vce(robust)

*here is where we create the residuals so we can graph them. They come from the regression directly above. We cannot use this when aggregate is in use.
predict resid, residuals
//Wild Bootstrap//
xtdidregress (retention adm_rate cost_attend avgfacsal urate gdp med_hs_income faminc ownership pell_ever) (legal), group(fips) time(year) aggregate(standard)
xtdidregress (retention adm_rate cost_attend avgfacsal urate gdp med_hs_income faminc ownership pell_ever) (legal), group(fips) time(year) aggregate(dlang)


*if we want to go back to the graduation rate we can change the gen to be the mean graduation rate and then run the same code
//Create averages for simpler graphing//
egen control = mean (resid)  if treated == 0 , by (year)
egen treated_2013 = mean (resid) if treated == 2013, by (year)
egen treated_2015 = mean (resid) if treated == 2015, by (year)
egen treated_2017 = mean (resid) if treated == 2017, by (year)
egen treated_2019 = mean (resid) if treated == 2019, by (year)

//Sort so the graph looks clean//
sort year

*the graph still has a lot of noise even when we just graph the residuals
//Create graph centered on treatement time//
twoway (line control years_before) (line treated_2013 years_before) (line treated_2015 years_before) (line treated_2017 years_before) (line treated_2019 years_before), xline(0) title("Retention Rates by State") ytitle("Retention Rates") xtitle("Years")


//Create graph with lines at treatment times and labels//
twoway (line control year) (line treated_2013 year) (line treated_2015 year) (line treated_2017 year) (line treated_2019 year), xline(2013) xline(2015) xline(2017) xline(2019) title("Retention Rates by State") ytitle("Retention Rates") xtitle("Years")


//Specific treated groups//
//2013//
twoway (line control years_before) (line treated_2013 years_before) , xline(0) title("Retention Rates by State") ytitle("Retention Rates") xtitle("Years")

twoway (line control year) (line treated_2013 year) , xline(2013) title("Retention Rates by State") ytitle("Retention Rates") xtitle("Years")

//2015//
twoway (line control years_before) (line treated_2015 years_before) , xline(0) title("Retention Rates by State") ytitle("Retention Rates") xtitle("Years")

twoway (line control year) (line treated_2015 year) , xline(2015) title("Retention Rates by State") ytitle("Retention Rates") xtitle("Years")

//2017//
twoway (line control years_before) (line treated_2017 years_before) , xline(0) title("Retention Rates by State") ytitle("Retention Rates") xtitle("Years")

twoway (line control year) (line treated_2017 year) , xline(2017) title("Retention Rates by State") ytitle("Retention Rates") xtitle("Years")

//2019//
twoway (line control years_before) (line treated_2019 years_before) , xline(0) title("Retention Rates by State") ytitle("Retention Rates") xtitle("Years")

twoway (line control year) (line treated_2019 year) , xline(2019) title("Retention Rates by State") ytitle("Retention Rates") xtitle("Years")


//Save and Export graph for future use//
*graph export "C:\Users\hyattjon\Box\Econ 398\Final Project\parallel_trends.jpg", as(jpg) quality(90) replace
*graph save "C:\Users\hyattjon\Box\Econ 398\Final Project\parallel_trends.gph", replace

//Close the log//
log close


*change the variables from strings and drop the privacy stuff
*regress admin rate and replace missing values
*cost attend avgfacsal loan ever pell ever faminc maybe we only use faminc instead of pell or loan
