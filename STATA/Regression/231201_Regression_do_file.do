cap log close
//Change the directory for the duration of the do file//
//Use this when on Physical Computer//
*cd "C:\Users\hyattjon.BYU\Box\Econ 398\Final Project"
//Use when on Cloud Drive//
cd "C:\Users\hyunsung.BYU\Box\Final Project\data\"

//Start the log file//
log using "Final Project.smcl",replace

//Import Final Dataset//
use final_data.dta, clear

//Drop NULL Value//
drop if retention == "NULL"
destring retention, replace

//Organize Dataset so it's easier to read//
sort fips instnm year

//Create variable for treatment and which year the state was treated//
gen treated = 0
replace treated = 2012 if fips == 8 | fips == 53
replace treated = 2014 if fips == 2 | fips == 11 | fips == 41
replace treated = 2016 if fips == 6 | fips == 23 | fips == 32

//Add More Covariates//
replace adm_rate = "." if adm_rate == "NULL"
destring adm_rate, replace
replace cost_attend = "." if cost_attend == "NULL"
destring cost_attend, replace
replace avgfacsal = "." if avgfacsal == "NULL"
destring avgfacsal, replace

//Print out graphs used to check for balance//
sum if treat ==1
sum if treat ==0

//Setting variables for XTDIDRegression////
ssc install event_plot
encode instnm, generate(inst)
xtset fips

//Basic Regression//
xtdidregress (retention urate gdp totemploy) (legal), group(fips) time(year)
xtdidregress (retention urate gdp totemploy med_hs_income ownership) (legal), group(fips) time(year)
xtdidregress (retention urate gdp adm_rate cost_attend) (legal), group(fips) time(year)
*Just trying out different combination to bring P-value down. Lowest P-value we found so far is .505 with (urate gdp adm_rate cost_attend)

----------------------------------------------------------------------------------This is how far I got_Hyun Dec 01 1:15pm

//FOR REFERENCE : Jane's regression//
xi: regress retention legal i.inst i.year




//Robust standard errors//
xtdidregress (gr FP4year statepop interestrate inflationrate med_hs_income totemploy urate gdp) (legal), group(fips) time(year) vce(robust)
//Wild Bootstrap//
xtdidregress (gr FP4year statepop interestrate inflationrate med_hs_income totemploy urate gdp) (legal), group(fips) time(year) wildbootstrap
//Aggregate Data//
*We need to maybe look at using other data to fix this issue
xtdidregress (gr FP4year statepop interestrate inflationrate med_hs_income totemploy urate gdp) (legal), group(fips) time(year) aggregate(standard)
xtdidregress (gr FP4year statepop interestrate inflationrate med_hs_income totemploy urate gdp) (legal), group(fips) time(year) aggregate(dlang)





//Drop DC due to high variation in graduation rates//
drop if fips == 11

//Create averages for simpler graphing//
egen control = mean (gr)  if treated == 0 , by (year)
egen treated_2012 = mean (gr) if treated == 2012, by (year)
egen treated_2014 = mean (gr) if treated == 2014, by (year)
egen treated_2016 = mean (gr) if treated == 2016, by (year)


//Sort so the graph looks clean//
sort year

//Create graph with lines at treatment times and labels//
twoway (line control year) (line treated_2012 year) (line treated_2014 year) (line treated_2016 year), xline(2012) xline(2014) xline(2016) title("Graduation Rates by State") ytitle("Graduation Rates") xtitle("Years")

//Save and Export graph for future use//
graph export "C:\Users\hyattjon\Box\Econ 398\Final Project\parallel_trends.jpg", as(jpg) quality(90) replace
graph save "C:\Users\hyattjon\Box\Econ 398\Final Project\parallel_trends.gph", replace

//Close the log//
log close

* talk about how we can add robustness checks
* we dont have hetero effects