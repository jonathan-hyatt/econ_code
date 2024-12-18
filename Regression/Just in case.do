/*
//Create averages for simpler graphing//
foreach x in basic_resid adm_rate_resid robust_resid bootstrap_resid{
	egen control_`x' = mean (`x')  if treated == 0 , by (year)
	egen treated_2013_`x' = mean (`x') if treated == 2013, by (year)
	egen treated_2015_`x' = mean (`x') if treated == 2015, by (year)
	egen treated_2017_`x' = mean (`x') if treated == 2017, by (year)
	
	//Sort so the graph looks clean//
sort year


//Create graph centered on treatement time//
twoway (line control_`x' years_before) (line treated_2013_`x' years_before, lpattern(dash)) (line treated_2015_`x' years_before, lpattern(dash_dot)) (line treated_2017_`x' years_before, lpattern(longdash)), xline(0) title("Retention Rates by State") ytitle("Retention Rates") xtitle("Years")

*Save as JPG for Paper
graph export "C:\Users\hyattjon\Box\Econ 398\Final Project\graphs\parallel_trends_centered_`x'.jpg", as(jpg) quality(90) replace
*Save as gph
graph save "C:\Users\hyattjon\Box\Econ 398\Final Project\graphs\parallel_trends_centered_`x'.gph", replace

//Create graph with lines at treatment times and labels//
twoway (line control year) (line treated_2013 year, lpattern(dash)) (line treated_2015 year, lpattern(dash_dot)) (line treated_2017 year, lpattern(longdash)), xline(2013, lpattern(solid)) xline(2015, lpattern(solid)) xline(2017, lpattern(solid)) title("Retention Rates by State") ytitle("Retention Rates") xtitle("Years")

*Save as JPG for Paper
graph export "C:\Users\hyattjon\Box\Econ 398\Final Project\graphs\parallel_trends.jpg", as(jpg) quality(90) replace
*Save as gph
graph save "C:\Users\hyattjon\Box\Econ 398\Final Project\graphs\parallel_trends.gph", replace

//Specific treated groups//
//2013//
twoway (line control years_before) (line treated_2013 years_before, lpattern(dash)) , xline(0, lpattern(solid)) title("Retention Rates by State") ytitle("Retention Rates") xtitle("Years")

*Save as JPG for Paper
graph export "C:\Users\hyattjon\Box\Econ 398\Final Project\graphs\2013_centered.jpg", as(jpg) quality(90) replace
*Save as gph
graph save "C:\Users\hyattjon\Box\Econ 398\Final Project\graphs\2013_centered.gph", replace

twoway (line control year) (line treated_2013 year, lpattern(dash)) , xline(2013, lpattern(solid)) title("Retention Rates by State") ytitle("Retention Rates") xtitle("Years")

*Save as JPG for Paper
graph export "C:\Users\hyattjon\Box\Econ 398\Final Project\graphs\2013_basic.jpg", as(jpg) quality(90) replace
*Save as gph
graph save "C:\Users\hyattjon\Box\Econ 398\Final Project\graphs\2013_basic.gph", replace

//2015//
twoway (line control years_before) (line treated_2015 years_before, lpattern(dash)) , xline(0) title("Retention Rates by State") ytitle("Retention Rates") xtitle("Years")

*Save as JPG for Paper
graph export "C:\Users\hyattjon\Box\Econ 398\Final Project\graphs\2015_centered.jpg", as(jpg) quality(90) replace
*Save as gph
graph save "C:\Users\hyattjon\Box\Econ 398\Final Project\graphs\2015_centered.gph", replace

twoway (line control year) (line treated_2015 year, lpattern(dash)) , xline(2015) title("Retention Rates by State") ytitle("Retention Rates") xtitle("Years")

*Save as JPG for Paper
graph export "C:\Users\hyattjon\Box\Econ 398\Final Project\graphs\2015_basic.jpg", as(jpg) quality(90) replace
*Save as gph
graph save "C:\Users\hyattjon\Box\Econ 398\Final Project\graphs\2015_basic.gph", replace

//2017//
twoway (line control years_before) (line treated_2017 years_before, lpattern(dash)) , xline(0) title("Retention Rates by State") ytitle("Retention Rates") xtitle("Years")

*Save as JPG for Paper
graph export "C:\Users\hyattjon\Box\Econ 398\Final Project\graphs\2017_centered.jpg", as(jpg) quality(90) replace
*Save as gph
graph save "C:\Users\hyattjon\Box\Econ 398\Final Project\graphs\2017_centered.gph", replace

twoway (line control year) (line treated_2017 year, lpattern(dash)) , xline(2017) title("Retention Rates by State") ytitle("Retention Rates") xtitle("Years")

*Save as JPG for Paper
graph export "C:\Users\hyattjon\Box\Econ 398\Final Project\graphs\2017_basic.jpg", as(jpg) quality(90) replace
*Save as gph
graph save "C:\Users\hyattjon\Box\Econ 398\Final Project\graphs\2017_basic.gph", replace

}
*/

//Sort so the graph looks clean//
sort year


//Create graph centered on treatement time//
twoway (line control years_before) (line treated_2013 years_before, lpattern(dash)) (line treated_2015 years_before, lpattern(dash_dot)) (line treated_2017 years_before, lpattern(longdash)), xline(0) title("Retention Rates by State") ytitle("Retention Rates") xtitle("Years")

*Save as JPG for Paper
graph export "C:\Users\hyattjon\Box\Econ 398\Final Project\graphs\parallel_trends_centered.jpg", as(jpg) quality(90) replace
*Save as gph
graph save "C:\Users\hyattjon\Box\Econ 398\Final Project\graphs\parallel_trends_centered.gph", replace

//Create graph with lines at treatment times and labels//
twoway (line control year) (line treated_2013 year, lpattern(dash)) (line treated_2015 year, lpattern(dash_dot)) (line treated_2017 year, lpattern(longdash)), xline(2013, lpattern(solid)) xline(2015, lpattern(solid)) xline(2017, lpattern(solid)) title("Retention Rates by State") ytitle("Retention Rates") xtitle("Years")

*Save as JPG for Paper
graph export "C:\Users\hyattjon\Box\Econ 398\Final Project\graphs\parallel_trends.jpg", as(jpg) quality(90) replace
*Save as gph
graph save "C:\Users\hyattjon\Box\Econ 398\Final Project\graphs\parallel_trends.gph", replace