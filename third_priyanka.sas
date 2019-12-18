libname pg1 base "C:\Schulich\Transfer to Asus\Schulich MBAN D\Fall Term\Predictive Modelling II\Assignment 1";
proc print data=pg1.hist2(obs=10);
run;
* Run summary statitistics to check missing values, mean, sum, median;
proc means data=pg1.hist2 n nmiss mean sum median;
var Recency Frequency Seniority TotalGift MinGift MaxGift;
run;
* Set population proportion;
%let pi1=0.02;
data hist2;
	 set pg1.hist2;
	 * Assign numerical values to the 3 categories;
	 if Education = "Elementary" then Edu_level=1;
	 else if Education = "High School" then Edu_level = 2; 
	 else if Education = "University / College" then Edu_level =3;

	 *Assign numerical values to categories. This is not dummy variable coding;
	if City = "Downtown" then Res =1;
   	else 
	if City = "Suburban" then Res =2;
	else 
	if City = "Rural" then Res=3;
	else Res =4;

	*Impute missing values with median of the distribution of each variable;
		if recency=. then recency =0;
		if recency = . then recency =2;
		if frequency = . then frequency = 1;
		if seniority = . then seniority = 4;
		if totalgift = . then totalgift = 40;
		if mingift = . then mingift = 20;
		if maxgift = . then maxgift = 30;

	drop Education City;
run;
*Check the correlation with target variable. The 'rank' keyword also givies the level of impact to the target variable, with the highest impact on the left to the least impacyful variable to the right;
proc corr data=hist2 rank
	plots(only)=scatter(nvar=all ellipse=none);
	var woman age salary Edu_level Res seniorlist NbActivities referrals recency frequency seniority totalgift mingift maxgift gavelastyear amtlastyear contact;
	with gavethisyear;
	id ID;
run;
*Check for collinearity;
proc corr data=hist2;
	ods output PearsonCorr=P;
	*plots(maxpoints=none only)=scatter(nvar=all ellipse=none);
	var woman age salary Edu_level Res seniorlist NbActivities referrals recency frequency seniority totalgift mingift maxgift gavelastyear amtlastyear contact;
	
run;
*Define macro variables;
%let cat_vars = woman Edu_level Res gavelastyear contact;
%let ind_vars = age salary seniorlist NbActivities referrals recency frequency seniority totalgift mingift maxgift amtlastyear;
*Split imputed dataset into &0% training data and 30% validation data;
proc surveyselect data=hist2 out=hist2_split method=srs samprate=0.70
  outall seed=12345 noprint;
  samplingunit id;
run;
data train;
	 set hist2_split;
	 if Selected=1;
run;
data validation;
	  set hist2_split;
	  if Selected=0;
run;
ods select roccurve;
proc logistic data=train;
class woman(param=ref ref='0') Edu_level(param=ref ref='3') Res(param=ref ref='4') gavelastyear contact;
model gavethisyear(event='1')=&cat_vars &ind_vars
/ stb clodds=pl;
score data=validation priorevent=0.02 outroc=roc;
run;
data roc1;
set roc;
cutoff=_PROB_;
specif=1-_1MSPEC_;
tp=&pi1*_SENSIT_;
fn=&pi1*(1-_SENSIT_);
tn=(1-&pi1)*specif;
fp=(1-&pi1)*_1MSPEC_;
acc=tp+tn;
depth=tp+fp;
pospv=tp/depth;
negpv=tn/(1-depth);
lift=pospv/&pi1;
keep cutoff tn fp fn tp _SENSIT_ _1MSPEC_
specif depth pospv negpv acc lift;
run;
*From this plot, we will always receive better operating surplus if we pick top 10% of the data---- that is why I selected 100,000 instead of 60,000;
proc sgplot data= roc1;
where 0.005 <= depth <= 0.50;
series y=lift x=depth;
refline 1.0 / axis=y;
yaxis values=(0 to 9 by 1);
run;
quit;
*Build logistic model with no interaction term;
*The concordance percent was approx 70 and discordance percent was approx 30;
proc logistic data=train noprint;
class woman(param=ref ref='0') Edu_level(param=ref ref='3') Res(param=ref ref='4') gavelastyear contact;
model gavethisyear(event='1')=&cat_vars &ind_vars;
score data=validation out=validated1(rename=(p_1=p_1_beg));
run;

title1 'Determine P-Value for Entry and Retention';
*Using this step we will determine the p-value(s1) for entering or leaving the model;
proc sql;
select 1 - probchi(log(sum(gavethisyear ge 0)),1) into :sl
from train;
quit;
*Next build logistic regression with interaction terms(2-way) first using forward variable selection;
proc logistic data=train;
class woman(param=ref ref='0') Edu_level(param=ref ref='3') Res(param=ref ref='4') gavelastyear contact;
model gavethisyear(event='1')= &cat_vars &ind_vars
contact|nbactivities|gavelastyear|amtlastyear|maxgift|age|woman|recency|res|salary @2 / include=10 clodds=pl selection=forward slstay=&sl;
run;
*Assign macro variable with the variables selected in the above step;
%let screened = Woman Edu_Level Res GaveLastYear Contact Age Salary SeniorList NbActivities Referrals Recency Frequency Seniority TotalGift MinGift MaxGift AmtLastYear NbActivities*Contact GaveLastYear*Contact NbActivities*GaveLastYear MaxGift*Contact NbActivities*MaxGift Age*Contact Woman*Contact NbActivities*Woman NbActivities*Recency Recency*GaveLastYear Recency*AmtLastYear Res*Contact NbActivities*Res Res*GaveLastYear Age*Res Recency*Res Salary*Contact Salary*NbActivities Salary*Woman;
*Build logistic regression model with the screened variables but using backward fast to eliminate unnecessary terms, using s1 as the criteria for leaving;
proc logistic data=train;
class woman(param=ref ref='0') Edu_level(param=ref ref='3') Res(param=ref ref='4') gavelastyear contact;
model gavethisyear(event='1')= &screened /clodds=pl selection=backward fast slstay=&sl hier=single;
run;
*The backward method eliminates a few terms, leaving behind the following selected list;
%let selected = Woman Edu_Level Res GaveLastYear Contact Age Salary SeniorList NbActivities Referrals Recency Frequency Seniority TotalGift MaxGift NbActivities*Contact GaveLastYear*Contact NbActivities*GaveLastYear NbActivities*MaxGift Age*Contact Woman*Contact NbActivities*Recency Recency*GaveLastYear Res*Contact NbActivities*Res Recency*Res Salary*Contact Salary*Woman;
proc logistic data=train noprint;
model gavethisyear(event='1')=&selected;
score data=validation out=validated2(rename=(p_1=p_screened));
run;
*Using the selected variables build logistic regression. Higher concordance percentage;
proc sort data=validated1;
by ID;
run;
proc sort data=validated2;
by ID;
run;
data validated;
	  merge validated1 validated2;
	  by ID;
run;
* Compare the c-static of the two logistic regression model nuilt, one without interaction terms and one with interaction terms. This test shows the one with interaction terms performed better;
ods select ROCOverlay ROCAssociation ROCContrastTest;
title1 "Validation Data Set Performance";
proc logistic data=validated;
model gavethisyear(event='1')=p_1_beg p_screened/nofit;
roc "Model with only main effects" p_1_beg;
roc "Model with interaction terms" p_screened;
roccontrast "Comparing the Two Models";
run;
*model with interaction terms is significantly better than the model without the terms hence, this is a better model.;



*******Regression Model;
title1 'Assess Target Normality';
*Check for noramlity of data;
proc univariate data=hist2;
var AmtThisYear;
histogram AmtThisYear /normal(mu=est sigma=est);
*probplot AmtThisYear /normal(mu=est sigma=est);
where GaveThisYear=1;
run;
*Create variable clusters to reduce collinearity;
proc varclus data=train maxeigen=0.7 hi 
	  outtree=tree;
	  var &cat_vars &ind_vars;
run;
** Cluster 1- nbactivities cluster 2- maxgift cluster 3 - recency;
%let cat_vars1 = woman Edu_level Res gavelastyear contact;
%let ind_vars1 = age salary seniorlist NbActivities recency maxgift amtlastyear;
ods select none;
ods output spearmancorr=spearman hoeffdingcorr=hoeffding;

proc corr data=hist2 spearman hoeffding;
var GaveThisYear;
with &cat_vars1 &ind_vars1;
run;
ods select all;
proc sort data=spearman;
by variable;
run;
proc sort data=hoeffding;
by variable;
run;

data temp;
merge spearman(rename=(gavethisyear=scorr pgavethisyear=spvalue))
hoeffding(rename=(gavethisyear=hcorr pgavethisyear=hpvalue));
by variable;
scorr_abs=abs(scorr);
hcorr_abs=abs(hcorr);
run;
proc rank data=temp out=correlations descending;
var scorr_abs hcorr_abs;
ranks ranksp rankho;
run;
proc sort data=correlations;
by ranksp;
run;
title1 "Rank of Spearman Correlations and Hoeffding Correlations";
proc print data=correlations label split='*';
var variable ranksp rankho scorr spvalue hcorr hpvalue;
label ranksp ='Spearman rank*of variables'
scorr ='Spearman Correlation'
spvalue='Spearman p-value'
rankho ='Hoeffding rank*of variables'
hcorr ='Hoeffding Correlation'
hpvalue='Hoeffding p-value';
run;

proc sql;
select min(ranksp) into :vref
from (select ranksp
from correlations
having spvalue > 0.001);
select min(rankho) into :href
from (select rankho
from correlations
having hpvalue > 0.5);
quit;
*From this graph, none of the variables should be dropped. They are all below the reference lines.;
proc sgplot data=correlations;
refline &vref / axis=y;
refline &href / axis=x;
scatter y=ranksp x=rankho / datalabel=variable;
yaxis label="Rank of Spearman";
xaxis label="Rank of Hoeffding";
run;
*%let cat_vars1 = woman Edu_level Res gavelastyear contact;
*%let ind_vars1 = age salary seniorlist NbActivities recency maxgift amtlastyear;
*Select only people who gave this year;
data reg_data;
	 set hist2;
	 where GaveThisYear=1;
run;
*Randomly split the group of people who gave this year into 70-30 split;
proc surveyselect data=reg_data out=reg_data_split method=srs samprate=0.70
  outall seed=12345 noprint;
  samplingunit id;
run;
data train_reg_data;
	 set reg_data_split;
	 if Selected=1;
run;

data val_reg_data;
	 set reg_data_split;
	 if Selected=0;
run;
ods graphics on;
title1 'AIC';
%let cat_vars2 = woman Edu_level Res gavelastyear contact;
*Build multiple regression models using training data and test on validation data and compare MSE. Model modelop1, had the least MSE and hence this was selected for final phase;
proc glmselect data=train_reg_data plots=all;*Adj R-square: 0.0149 R-square:0.0152 MSE:49927;
class woman(param=ref ref='0') Edu_level(param=ref ref='3') Res(param=ref ref='4') gavelastyear contact;
AIC: model AmtThisYear=&cat_vars2 &ind_vars1 contact|nbactivities|gavelastyear|amtlastyear|maxgift|age|woman|recency|res|salary @2 / 
selection=stepwise select=AIC details=steps showpvalues;
store out=modelop;
run;
proc plm restore=modelop;
	 score data=val_reg_data out=val_reg_data_done;
	 show fit parms;
run;
***Best regression model with R-square = 0.0154;
proc glmselect data=train_reg_data plots=all;*Adj R-square: 0.0151 R-square:0.0155 MSE:49915;
class woman(param=ref ref='0') Edu_level(param=ref ref='3') Res(param=ref ref='4') gavelastyear contact;
AIC: model AmtThisYear=&cat_vars2 &ind_vars contact|nbactivities|gavelastyear|amtlastyear|maxgift|age|woman|recency|res|salary|Edu_Level @2 / 
selection=stepwise select=AIC details=steps showpvalues;
store out=modelop1;
run;
proc plm restore=modelop1;
	 score data=val_reg_data out=val_reg_data_done;
	 show fit parms;
run;
proc glmselect data=train_reg_data plots=all;*Adj R-square: 0.0143 R-square:0.0141 MSE:49965;
class woman(param=ref ref='0') Edu_level(param=ref ref='3') Res(param=ref ref='4') gavelastyear contact;
SBC: model AmtThisYear=&cat_vars2 &ind_vars contact|nbactivities|gavelastyear|amtlastyear|maxgift|age|woman|recency|res|salary|Edu_Level @2 / 
selection=stepwise select=SBC details=steps showpvalues;
store out=modelop2;
run;
proc plm restore=modelop2;
	 score data=val_reg_data out=val_reg_data_done;
	 show fit parms;
run;

%let cat_vars3 = woman Res gavelastyear contact;
proc glmselect data=train_reg_data plots=all;*Adj R-square: 0.0152 R-square:0.0148 MSE:49930;
class woman(param=ref ref='0') Res(param=ref ref='4') gavelastyear contact;
AIC: model AmtThisYear=&cat_vars3 &ind_vars1 contact|nbactivities|gavelastyear|amtlastyear|maxgift|age|woman|recency|res|salary @2 / 
selection=stepwise select=AIC details=steps showpvalues;
store out=modelop3;
run;
proc plm restore=modelop3;
	 score data=val_reg_data out=val_reg_data_done;
	 show fit parms;
run;
proc glmselect data=train_reg_data plots=all;*Adj R-square: 0.0153 R-square:0.0149 MSE:49925;
class woman(param=ref ref='0') Edu_level(param=ref ref='3') Res(param=ref ref='4') gavelastyear contact;
AIC: model AmtThisYear=&cat_vars2 &ind_vars1 contact|nbactivities|gavelastyear|amtlastyear|maxgift|age|woman|recency|res|salary @2 / 
selection=backward select=AIC details=steps showpvalues;
store out=modelop4;
run;
proc plm restore=modelop4;
	 score data=val_reg_data out=val_reg_data_done;
	 show fit parms;
run;
proc glmselect data=train_reg_data plots=all;*Adj R-square: 0.0152 R-square:0.0149 MSE:49927;
class woman(param=ref ref='0') Edu_level(param=ref ref='3') Res(param=ref ref='4') gavelastyear contact;
AIC: model AmtThisYear=&cat_vars2 &ind_vars1 contact|nbactivities|gavelastyear|amtlastyear|maxgift|age|woman|recency|res|salary @2 / 
selection=forward select=AIC details=steps showpvalues;
store out=modelop5;
run;
proc plm restore=modelop5;
	 score data=val_reg_data out=val_reg_data_done;
	 show fit parms;
run;
* Stepwise model is the best


****** Scoring*********;
*Prepare the scoring dataset the same the training dataset was prepared;
data score2_contact;
	 set pg1.score2_contact;
	 if Education = "Elementary" then Edu_level=1;
	 else if Education = "High School" then Edu_level = 2; 
	 else if Education = "University / College" then Edu_level =3;

	if City = "Downtown" then Res =1;
   	else 
	if City = "Suburban" then Res =2;
	else 
	if City = "Rural" then Res=3;
	else Res =4;
		if recency=. then recency =0;
		if recency = . then recency =2;
		if frequency = . then frequency = 1;
		if seniority = . then seniority = 4;
		if totalgift = . then totalgift = 40;
		if mingift = . then mingift = 20;
		if maxgift = . then maxgift = 30;



	drop Education City;
run;

data score2_nocontact;
	 set pg1.score2_nocontact;
	 if Education = "Elementary" then Edu_level=1;
	 else if Education = "High School" then Edu_level = 2; 
	 else if Education = "University / College" then Edu_level =3;

	if City = "Downtown" then Res =1;
   	else 
	if City = "Suburban" then Res =2;
	else 
	if City = "Rural" then Res=3;
	else Res =4;
		if recency=. then recency =0;
		if recency = . then recency =2;
		if frequency = . then frequency = 1;
		if seniority = . then seniority = 4;
		if totalgift = . then totalgift = 40;
		if mingift = . then mingift = 20;
		if maxgift = . then maxgift = 30;

	
	drop Education City;
run;
%let selected = Woman Edu_Level Res GaveLastYear Contact Age Salary SeniorList NbActivities Referrals Recency Frequency Seniority TotalGift MaxGift NbActivities*Contact GaveLastYear*Contact NbActivities*GaveLastYear NbActivities*MaxGift Age*Contact Woman*Contact NbActivities*Recency Recency*GaveLastYear Res*Contact NbActivities*Res Recency*Res Salary*Contact Salary*Woman;
proc logistic data=hist2;
model gavethisyear(event='1')=&selected;
score data=score2_contact out=scored_contact;
run;

proc plm restore=modelop1;
	 score data=score2_contact out=scored_contact_amt;
run;

proc logistic data=hist2;
model gavethisyear(event='1')=&selected;
score data=score2_nocontact out=scored_nocontact;
run;
proc plm restore=modelop1;
	 score data=score2_nocontact out=scored_nocontact_amt;
run;

proc sort data=scored_nocontact_amt out=scored_nocontact_amt_sorted(rename=(Predicted=Predicted_nocontact));
	 by descending ID;
run;
proc sort data=scored_contact_amt out=scored_contact_amt_sorted(rename=(Predicted=Predicted_contact));
	 by descending ID;
run;
proc sort data=scored_contact out=scored_contact_sorted(rename=(P_1=P_1_contact));
	 by descending ID;
run;
proc sort data=scored_nocontact out=scored_nocontact_sorted(rename=(P_1=P_1_nocontact));
	 by descending ID;
run;
data merged_table;
	 merge scored_nocontact_amt_sorted scored_contact_amt_sorted scored_nocontact_sorted scored_contact_sorted;
	 by descending ID;
	 EC = P_1_contact*Predicted_contact;
	 ENC = P_1_nocontact * Predicted_nocontact;
	 uplift = EC-ENC;

run;
proc sort data=merged_table out=merged_table_sorted;
	 by descending uplift;
run;
data first_10p;
	 set merged_table_sorted; *selecting top 10% of the dataset;
	 if uplift>30;
	 keep ID;
run;
proc means data=first_10p n sum;
var uplift;
run;
proc export data=first_10p outfile="C:\Schulich\Transfer to Asus\Schulich MBAN D\Fall Term\Predictive Modelling II\Assignment 1\team2p11_new3.csv"
dbms=csv;
run;

***fFor checking mean;
data first_10p;
	 set merged_table_sorted(obs=100000); *selecting top 10% of the dataset;
	 *if uplift>1;
	 *keep ID;
run;
proc means data=first_10p n sum;
var uplift;
run;
data pg2.team2part1;
	 set first_10p;
run;

proc print data=scored_contact(obs=10);
run;







