import numpy as np
import matplotlib.pyplot as plt
import csv
import random as rand
import math
import scipy.stats as stats
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata
import plotly.graph_objects as go
import streamlit as st



#This code is a model of all non statory work to determine wait times dependant on referral numbers and age distributions as well as diagnostic distributions. Capacity is calculated annually as it scales linearly with time similarly to demand. '''

##Global default Variables. diagnosis rates for each age group <6 and >6 were calculated form the 24/25 referral sheet. This covered 88% of referrals with the remainder being referrrals##
#referral rate
Total_ref_num=700
num_of_years=1

#staffing of service WTE
num_docs=5.1 #WTE doctors for comm paeds
num_nurses=2.4 #WTE nurses for comm paeds

#provide appt numbers and types at a service or individual level to initiate the fu and new capacity per clinic variables
total_clinic_num_doc=4#total number of non-staturory clinics over a set period for the doctors. Can be a week or month
total_new_appt_doc=5#total number of new appts over a set period for the doctors. Can be a week or month. Must be same as total clinic num
total_fu_appt_doc=20#total number of new appts over a set period for the doctors. Can be a week or month. Must be same as total clinic num
new_per_clinic_doc= total_new_appt_doc/total_clinic_num_doc#no new patient appt per clinic
fu_per_clinic_doc= total_fu_appt_doc/total_clinic_num_doc #no of f/u appt per clinic
total_clinic_num_nurse=3#total number of non-staturory clinics over a set period for the nurses. Can be a week or month
total_new_appt_nurse=0.1#total number of new appts over a set period for the nurses. Can be a week or month. Must be same as total clinic num
total_fu_appt_nurse=12#total number of new appts over a set period for the nurses. Can be a week or month. Must be same as total clinic num
new_per_clinic_nurse= total_new_appt_nurse/total_clinic_num_nurse#no new patient appt per clinic
fu_per_clinic_nurse= total_fu_appt_nurse/total_clinic_num_nurse #no of f/u appt per clinic

#provide number of non statutory clinics that service provides in a week per clinician
num_of_clinics_docs = 3.5#per week per WTE doc
num_of_clinics_nurses = 3.5#per week per WTE nurse

#provide the diagnostic dependant fu burden. This encompasses the type of service model (1 stop shop, streamed) as well as discharge rates post assessment. ie not all ADHDs need f/u 
ADHD_fu_num=2#number of follow ups always needed post new assessment
ADHD_reg_fu_num=2#number of follow ups needed per year normally for each patient. This encompasess discharge rates post assessment
ASD_fu_num=2.5#number of follow ups always needed post new assessment
ASD_reg_fu_num=0.2#number of follow ups needed per year normally
Complex_fu_num=3.5#number of follow ups always needed post new assessment
Complex_reg_fu_num=1.5#number of follow ups needed per year normally
min_age=1.5
max_age=17.5
fu_DNA_rate=0.05
new_DNA_rate=0.01

default_vals={'Total_ref_num':Total_ref_num,'num_of_years':num_of_years,
'num_docs':num_docs,
'num_nurses':num_nurses,
'new_per_clinic_doc':new_per_clinic_doc, #number of average of new patient appt
'fu_per_clinic_doc':fu_per_clinic_doc, #number of average of f/u appt
'new_per_clinic_nurse':new_per_clinic_nurse,
'fu_per_clinic_nurse':fu_per_clinic_nurse,
'num_of_clinics_docs':num_of_clinics_docs,
'num_of_clinics_nurses':num_of_clinics_nurses,
'ADHD_fu_num':ADHD_fu_num,
'ADHD_reg_fu_num':ADHD_reg_fu_num,
'ASD_fu_num':ASD_fu_num,
'ASD_reg_fu_num':ASD_reg_fu_num,
'Complex_fu_num':Complex_fu_num,
'Complex_reg_fu_num':Complex_reg_fu_num,
'min_age':min_age,
'max_age':max_age,
'fu_DNA_rate':fu_DNA_rate,
'new_DNA_rate':new_DNA_rate}

#Hard coded variables in functions(cannot alter them here. These were derived from 23/24 referral data)
_vars='''Working_weeks=42
ASD_frac_under6=0.74
ADHD_frac_under6= 0.04
Complex_frac_under6=0.22
ASD_frac_over6=0.27
ADHD_frac_over6= 0.43
Complex_frac_over6=0.3'''

num_docs_range=np.arange(2,9,0.5)
num_nurses_range=np.arange(0,7,0.5)
ref_rate_range=np.arange(300,1550,50)
new_appt_type_range=np.arange(0.2,5,0.2)
fu_appt_type_range=np.arange(0.2,9,0.2)
fu_range=np.arange(0.2,4.2,0.2)
reg_fu_range=np.arange(0.2,4.2,0.2)


#functions
#model the age distribution. This model is based on observations of the actual distribution of age at referral over the last 3 years using a skewed normal fit
def find_age_dist(Total_ref_num,min_age,max_age,delay=1,plot=False,verbose=False):
    '''delay here refers to teh fact that teh distrubution of age when seen will be delayed by the wait time and is hardcoded. This is modelled simply as adding to the peak. default is 1'''
    age_data= np.genfromtxt('ages.csv',delimiter=',',skip_header=1)
    Total_ref_num=int(Total_ref_num)
    skew,peak_age,peak_sd=stats.skewnorm.fit(age_data)
    if verbose:
        print(f'fitted stats are skew {skew}, peak age {peak_age}, sd {peak_sd}')
    age_dist=stats.skewnorm.rvs(a=skew, loc=peak_age+delay, scale=peak_sd ,size=Total_ref_num)
    age_dist=np.clip(age_dist,min_age,max_age)
    if plot:
        x=np.linspace(min(age_data),max(age_data),1000)
        pdf=stats.skewnorm.pdf(x,skew,peak_age,peak_sd)
        plt.hist(age_data, bins=18, density=True, alpha=0.4, color='blue',label='Histogram of actual age distribution')
        plt.hist(age_dist, bins=18, density=True, alpha=0.4, color='red',label='Histogram of simulated age distribution')
        plt.plot(x,pdf,'r-',lw=2, label='Fitted skew normal pdf')
        plt.xlabel("Years")
        plt.ylabel("Density")
        plt.title("Skewed Normal Distribution")
        plt.legend()
        plt.show()
    return(age_dist)
#print(age_dist)
#print((ADHD_fu_num+(18-mean_wait-age_dist[age_dist>6])*ADHD_reg_fu_num)*ADHD_frac_over6)
#Calculating f/u burden over lifetime of paediatric care. This is dependant on diagnosis and age distribution only and provides different diagnostic rates dependant on age cutoff of 6 years(simplified)

def get_ADHD_fu_burden(age_dist,ADHD_fu_num,ADHD_reg_fu_num,ADHD_frac_under6=0.04,ADHD_frac_over6=0.43):
    '''fractions are hardcoded'''
    ADHD_follow_ups = np.sum((ADHD_fu_num+(18-age_dist[age_dist<6])*ADHD_reg_fu_num)*ADHD_frac_under6)+np.sum((ADHD_fu_num+(18-age_dist[age_dist>6])*ADHD_reg_fu_num)*ADHD_frac_over6)#total num of follw ups needed for lifetime of care
    return(ADHD_follow_ups)

def get_ASD_fu_burden(age_dist,ASD_fu_num,ASD_reg_fu_num,ASD_frac_under6=0.74,ASD_frac_over6=0.27):
    '''fractions are hardcoded'''
    ASD_follow_ups = np.sum((ASD_fu_num+(18-age_dist[age_dist<6])*ASD_reg_fu_num)*ASD_frac_under6)+np.sum((ASD_fu_num+(18-age_dist[age_dist>6])*ASD_reg_fu_num)*ASD_frac_over6)#total num of follw ups needed for lifetime of care
    return(ASD_follow_ups)

def get_Complex_fu_burden(age_dist,Complex_fu_num,Complex_reg_fu_num,Complex_frac_under6=0.22,Complex_frac_over6=0.3):
    '''fractions are hardcoded'''
    Complex_follow_ups = np.sum((Complex_fu_num+(18-age_dist[age_dist<6])*Complex_reg_fu_num)*Complex_frac_under6)+np.sum((Complex_fu_num+(18-age_dist[age_dist>6])*Complex_reg_fu_num)*Complex_frac_over6)#total num of follw ups needed for lifetime of care
    return(Complex_follow_ups)

def get_annual_fu_burdens(age_dist,total_new,total_fu,fu_DNA_rate):
    'returns the mean annual fu burden per new patient and the sd of that value. Also returns the mean and upper new to f/u ratios given the age and diagnosis distributions'
    lifetime_ratio=total_fu/total_new# Over the lifetime of the new patient what is the number of follows for a population. Remember the n is the same for total fu and new as we modelled teh age distribution with a set scale.  This does not vary significantly with ref rate but is dependant on underlying diagnosis and age distribution
    median_annual_fu_burden=(np.median(lifetime_ratio/(18-age_dist)))*(1+fu_DNA_rate)#taking the mean annual burden over the lifetime introduces the range
    #sd_annual_fu_burden=np.std(lifetime_ratio/(18-age_dist))
    lower_quantile=np.quantile(lifetime_ratio/(18-age_dist),0.25)
    upper_quantile=np.quantile(lifetime_ratio/(18-age_dist),0.75)
    #print('fu burdens per new patient',median_annual_fu_burden,lower_quantile,upper_quantile)
    upper_lim=np.ceil(upper_quantile)
    lower_lim=np.ceil(lower_quantile)
    return(median_annual_fu_burden,lower_quantile,upper_quantile,upper_lim,lower_lim)

#print(f"the ideal ratio of new to follow up is 1:{int(upper_lim)} and minimum safe is 1:{int(lower_lim)}")

def get_annual_new_burden(Total_ref_num,num_of_years,new_DNA_rate):
    new_annual_burden= (Total_ref_num/num_of_years)*(1+new_DNA_rate)
    return(new_annual_burden)

#Calculate capacity
def fu_capacity_docs(num_docs,num_of_clinics_docs,fu_per_clinic_doc,Working_weeks=42):
    '''returns the capacity over a year'''
    fu_capacity=(num_docs*num_of_clinics_docs*fu_per_clinic_doc*Working_weeks)
    return(fu_capacity)

def fu_capacity_nurses(num_nurses,num_of_clinics_nurses,fu_per_clinic_nurse,Working_weeks=42):    
    fu_capacity=num_nurses*num_of_clinics_nurses*fu_per_clinic_nurse*Working_weeks#total capacity
    return(fu_capacity)

def new_capacity_docs(num_docs,num_of_clinics_docs,new_per_clinic_doc,Working_weeks=42):
    new_capacity=(num_docs*num_of_clinics_docs*new_per_clinic_doc*Working_weeks)
    return(new_capacity)

def new_capacity_nurses(num_nurses,num_of_clinics_nurses,new_per_clinic_nurse,Working_weeks=42):
    new_capacity=(num_nurses*num_of_clinics_nurses*new_per_clinic_nurse*Working_weeks)#total capacity
    return(new_capacity)

#calculate demand
def get_fu_demand(median_annual_fu_burden,lower_quantile,upper_quantile,annual_new_burden):
    '''gets the annual follow up burden from the calculated ratio of new to f/u'''
    fu_demand_mean=(median_annual_fu_burden)*annual_new_burden
    fu_demand_min=(lower_quantile)*annual_new_burden#using the sd of fu burden as a proxy as it takes into account diagnosis and age
    fu_demand_max=(upper_quantile)*annual_new_burden
    #print('min and max annual fu demand',fu_demand_max,fu_demand_min)
    return(fu_demand_max,fu_demand_min,fu_demand_mean)

def get_wait_times(fu_demand_min,fu_demand_max,fu_demand_mean,new_demand,fu_capacity,new_capacity):
    mean_wait_time=(fu_demand_mean/fu_capacity)+(new_demand/new_capacity)
    min_wait_time=(fu_demand_min/fu_capacity)+(new_demand/new_capacity)
    max_wait_time=(fu_demand_max/fu_capacity)+(new_demand/new_capacity)
    #print(f'The lower bound wait_time using a service workforce of {num_docs} docs and {num_nurses} nurses is {min_wait_time*12} months. The upper bound wait time is {max_wait_time*12} months')
    return(min_wait_time,max_wait_time,mean_wait_time)

def run_sim(Total_ref_num,num_of_years,num_docs,num_nurses,new_per_clinic_doc,fu_per_clinic_doc,new_per_clinic_nurse,fu_per_clinic_nurse,num_of_clinics_docs,num_of_clinics_nurses,
            ADHD_fu_num,ADHD_reg_fu_num,ASD_fu_num,ASD_reg_fu_num,Complex_fu_num,Complex_reg_fu_num,min_age,max_age,fu_DNA_rate,new_DNA_rate,verbose=False):
    '''runs the sim and allows the local variables in the argument to vary. Uses a dict of default vals'''
    age_dist=find_age_dist(Total_ref_num,min_age,max_age,plot=False)
    if verbose:
        print('Fitted age distribution')
    total_fu=get_ADHD_fu_burden(age_dist,ADHD_fu_num,ADHD_reg_fu_num)+get_ASD_fu_burden(age_dist,ASD_fu_num,ASD_reg_fu_num)+get_Complex_fu_burden(age_dist,Complex_fu_num,Complex_reg_fu_num)
    median_annual_fu_burden,lower_quantile,upper_quantile,upper_lim,lower_lim=get_annual_fu_burdens(age_dist,Total_ref_num,total_fu,fu_DNA_rate)
    #print(f"the ideal ratio of new to follow up is 1:{int(upper_lim)} and minimum safe is 1:{int(lower_lim)}")
    annual_new_burden=get_annual_new_burden(Total_ref_num,num_of_years,new_DNA_rate)
    if verbose:
        print('Got burdens, now calculating capacity')
    fu_capacity=fu_capacity_docs(num_docs,num_of_clinics_docs,fu_per_clinic_doc)+fu_capacity_nurses(num_nurses,num_of_clinics_nurses,fu_per_clinic_nurse)
    new_capacity=new_capacity_docs(num_docs,num_of_clinics_docs,new_per_clinic_doc)+new_capacity_nurses(num_nurses,num_of_clinics_nurses,new_per_clinic_nurse)
    #print('new capacity is ',new_capacity)
    if verbose:
        print('Got capacity, now calculating annual follow demand median and ranges')
    fu_demand_max,fu_demand_min,fu_demand_mean=get_fu_demand(median_annual_fu_burden,lower_quantile,upper_quantile,annual_new_burden)
    min_wait_time,max_wait_time,mean_wait_time=get_wait_times(fu_demand_min,fu_demand_max,fu_demand_mean,annual_new_burden,fu_capacity,new_capacity)
    results=np.array([min_wait_time*12,max_wait_time*12,mean_wait_time*12,Total_ref_num,num_docs,num_nurses])
    if verbose:
        print('Got follow up demand and generated results')
    results[results<0]=0

    return(results)

#########execute###########################################################################################################
options={'Referral Rates':False,'Workforce':False,
'New to F/U ratio for doctors':False,'Number of follow ups for conditions':False}#alterations here will be effective if streamlit is false. 
#run_streamlit=True
options = {key: False for key in options}#set default to No option selected

st.set_page_config(layout="wide")

st.title ("Community Paediatric Service Model to determine Waiting times")
st.write("This model allows you to adjust parameters for your service. Adjust the parameters below for your service. Aspects you choose to explore will be adjusted automatically")
col1, col2 = st.columns(2)  # Adjust the ratio for left/right width
with col1:
    with st.container():
        st.markdown('### Adjust these sliders for your service')
    # Create sliders to set default vals
        num_docs = st.slider("Select number of WTE docs", min_value=float(num_docs_range[0]), max_value=float(num_docs_range[-1]), step=0.1)
        default_vals['num_docs']=num_docs
        num_nurses = st.slider("Select number of WTE nurses", min_value=float(num_nurses_range[0]), max_value=float(num_nurses_range[-1]), step=0.1)
        default_vals['num_nurses']=num_nurses
        num_years=st.slider("Select number of years that referrals are counted", min_value=float(1), max_value=float(10), step=0.25)
        default_vals['num_of_years']=num_years
        ref_rate = st.slider("Select total number of referrals over that period of time", min_value=int(ref_rate_range[0]), max_value=int(ref_rate_range[-1]), step=int(50.0))
        default_vals['Total_ref_num']=ref_rate
        new_per_clinic_doc = st.slider("Select the number of New appointments per clinic per doctor on average", min_value=float(new_appt_type_range[0]), max_value=float(new_appt_type_range[-1]), step=0.1)
        default_vals['new_per_clinic_doc']=new_per_clinic_doc
        fu_per_clinic_doc = st.slider("Select the number of Follow up appointments per clinic per doctor on average", min_value=float(fu_appt_type_range[0]), max_value=float(fu_appt_type_range[-1]), step=0.1)
        default_vals['fu_per_clinic_doc']=fu_per_clinic_doc
        new_per_clinic_nurse = st.slider("Select the number of New appointments per clinic per Nurse on average", min_value=float(new_appt_type_range[0]), max_value=float(new_appt_type_range[-1]), step=0.1)
        default_vals['new_per_clinic_nurse']=new_per_clinic_nurse
        fu_per_clinic_nurse = st.slider("Select the number of Follow up appointments per clinic per Nurse on average", min_value=float(fu_appt_type_range[0]), max_value=float(fu_appt_type_range[-1]), step=0.1)
        default_vals['fu_per_clinic_nurse']=fu_per_clinic_nurse
        fu_appts_ADHD = st.slider("Select the number of Follow up appointments on average needed to diagnose ADHD", min_value=float(fu_range[0]), max_value=float(fu_range[-1]), step=0.1)
        default_vals['ADHD_fu_num']=fu_appts_ADHD
        reg_fu_appts_ADHD = st.slider("Select the number of Annual follow up appointments on average needed to manage ADHD", min_value=float(reg_fu_range[0]), max_value=float(reg_fu_range[-1]), step=0.1)
        default_vals['ADHD_fu_num']=fu_appts_ADHD   
        fu_appts_ASD = st.slider("Select the number of Follow up appointments on average needed to diagnose ASD", min_value=float(fu_range[0]), max_value=float(fu_range[-1]), step=0.1)
        default_vals['ASD_fu_num']=fu_appts_ASD
        reg_fu_appts_ASD = st.slider("Select the number of Annual follow up appointments on average needed to manage ASD", min_value=float(reg_fu_range[0]), max_value=float(reg_fu_range[-1]), step=0.1)
        default_vals['ASD_fu_num']=fu_appts_ASD  
        fu_appts_Complex = st.slider("Select the number of Follow up appointments on average needed to diagnose Complex disorders", min_value=float(fu_range[0]), max_value=float(fu_range[-1]), step=0.1)
        default_vals['Complex_fu_num']=fu_appts_Complex
        reg_fu_appts_Complex = st.slider("Select the number of Annual follow up appointments on average needed to manage Complex disorders", min_value=float(reg_fu_range[0]), max_value=float(reg_fu_range[-1]), step=0.1)
        default_vals['Complex_fu_num']=fu_appts_Complex 
        num_clinic_doc = st.slider("Select number of clinics per week on average a WTE doctor would do", min_value=float(0.1), max_value=float(10), step=0.1)
        default_vals['num_of_clinics_docs']=num_clinic_doc
        num_clinic_nurse = st.slider("Select number of clinics per week on average a WTE nurse would do", min_value=float(0.1), max_value=float(10), step=0.1)
        default_vals['num_of_clinics_nurses']=num_clinic_nurse
        fu_DNA_rate = st.slider("Select the fraction of follow up DNAs", min_value=float(0), max_value=float(0.2), step=0.01)
        default_vals['fu_DNA_rate']=fu_DNA_rate
        new_DNA_rate = st.slider("Select the fraction of New patient DNAs", min_value=float(0), max_value=float(0.2), step=0.01)
        default_vals['new_DNA_rate']=fu_DNA_rate

#Now choose what to explore

with col2:
    with st.container():
        st.markdown('### Choose what to explore and the plot apperas. (May take a few minutes)')
        selected_option = st.radio(
        "Now choose an aspect to explore. (You can only explore one at a time):",
        ["Referral Rates", "Workforce", "New to F/U ratio for doctors","Number of follow ups for conditions"]
        )
        st.write(f"You selected: {selected_option}. You can zoom in on the plots below")
        options[selected_option]=True


        explore_workforce=options['Workforce']
        explore_referral_rate=options['Referral Rates']#If both are false then a single service model is created
        explore_models=options['New to F/U ratio for doctors']
        explore_fu_rate=options['Number of follow ups for conditions']
        print(options)


        if explore_referral_rate:
            results=np.empty((0,6))
            for ref_num in ref_rate_range:
                params={**default_vals,'Total_ref_num':ref_num}
                result=run_sim(**params)
                results=np.vstack((results,result))
            for i,val in enumerate(results[:,3]):
                print(f"At a referral rate of {val}, the mean wait time is {results[i,2]} months with upper bound {results[i,1]} and lower bound {results[i,0]} months ' ")        
            plot_2d=True    
            if plot_2d:
                docs = results[:, 4]  
                nurses = results[:, 5]  
                min_time = results[:, 0]
                max_time = results[:, 1] 
                mean_time= results[:, 2]
                ref_rate=results[:,3]
                fig,ax=plt.subplots()
                #ax.errorbar(ref_rate,mean_time,yerr=[min_time,max_time],fmt='o',capsize=5, capthick=1, alpha=0.5,color='blue', label='Wait time with referral rate')
                ax.plot(ref_rate,mean_time,alpha=0.5,color='blue',label='Median')
                #ax.plot(ref_rate,max_time,alpha=0.2,color='red',label='Upper bound')
                #ax.plot(ref_rate,min_time,alpha=0.2,color='red',label='Lower bound')
                ax.fill_between(ref_rate, min_time, max_time, color="blue", alpha=0.2, label="Confidence interval")
                ax.set_xlabel("Referral Rate")
                ax.set_ylabel("Wait time (months)")
                ax.set_title("Wait time for current workforce at different referral rates")
                ax.axhline(y=12, color="red", linestyle="--", linewidth=1, label="12-Month Marker")
                ax.set_xlim(left=min(ref_rate)) 
                ax.legend()
                st.pyplot(fig)
        # Show the plot
                #plt.show()
                #fig.savefig("Ref_rate_plot.png", dpi=800, bbox_inches="tight")




        #explore the workforce range and plot        
        elif explore_workforce:
            results=np.empty((0,6))
            for i in num_docs_range:
                for j in num_nurses_range:
                    params={**default_vals,'num_docs':i,'num_nurses':j}
                    result=run_sim(**params)
                    results=np.vstack((results,result))           
            targ_wait=12# target wait time months
        # print('all results',results)
            filtered_results=results[results[:,2]<targ_wait]
            #print('filtered results',filtered_results)
            min_ref_rate=0
            min_docs=0
            min_nurses=0
            if filtered_results.size>0:
                min_idx=np.argmin(filtered_results[:,4])#filter by minimum number of doctors needed
                print(min_idx)
                min_ref_rate=filtered_results[min_idx,3]
                min_docs=filtered_results[min_idx,4]
                min_nurses=filtered_results[min_idx,5]

                print(f'for your service with an annual referral rate of {min_ref_rate} a minimum workforce of {min_docs} WTE doctors and {min_nurses} WTE nurses would keep mean wait time less than {targ_wait} months')
            else:
                print('No work force combination in that range can meet your wait target. extend the range or increase the target difference')
                min_idx=0

            plot_3d=True    
            if plot_3d:
                docs = results[:, 4]  
                nurses = results[:, 5]  
                min_time = results[:, 0]
                max_time = results[:, 1] 
                mean_time= results[:, 2]
                ref_rate=results[:,3]

                # Create plot
                fig = plt.figure(figsize=(10, 8))
                ax = fig.add_subplot(projection='3d')  # 3D plot

            

                # Define grid for the plane
                x_range = np.linspace(np.min(docs), np.max(docs),30)  # Adjust based on your data range
                y_range = np.linspace(np.min(nurses), np.max(nurses), 30)
                X, Y = np.meshgrid(x_range, y_range)

                Z = np.full_like(X, 12)  # Creates a flat plane at z = 12
                Z1= griddata((docs,nurses),mean_time,(X,Y),method='cubic')
                Z2= griddata((docs,nurses),min_time,(X,Y),method='cubic')
                Z3= griddata((docs,nurses),max_time,(X,Y),method='cubic')
                
                    # Create the plotly surface objects
                fig = go.Figure()

                fig.add_trace(go.Surface(x=X, y=Y, z=Z2, colorscale=[[0,'Red'],[1,'Red']], opacity=0.1, name='Lower Bound',showscale=False))
                fig.add_trace(go.Surface(x=X, y=Y, z=Z1, colorscale='Blues_r', opacity=1, name='Mean Time',showscale=True,colorbar=dict(title="Wait Time (months)")
            ))
                fig.add_trace(go.Surface(x=X, y=Y, z=Z3, colorscale=[[0,'Red'],[1,'Red']], opacity=0.1, name='Upper Bound',showscale=False))
                fig.add_trace(go.Surface(x=X, y=Y, z=Z, colorscale=[[0,'Gray'],[1,'Gray']], opacity=0.4, name='12 month boundary',showscale=False))

                # Update layout
                fig.update_layout(
                    title=f'Interpolated Wait Times for different workforce compositions at a referral rate of {ref_rate[0]}' ,
                    scene=dict(
                        xaxis_title='Doctors',
                        yaxis_title='Nurses',
                        zaxis_title='Wait Time (months)'
                    ),
                    annotations=[
                    dict(x=0, y=1.05 ,text=f'A minimum workforce of {min_docs} WTE doctors and {min_nurses}\n WTE nurses would keep mean wait time less than {targ_wait} months',showarrow=False, arrowhead=2)
                    ],
                    showlegend=True,
                    #width=900,
                    #height=900
                )
            
                col2.plotly_chart(fig, use_container_width=True)

                #fig.show()
                #fig.write_html("workforceplot.html")    
                    
        elif explore_models:
            #adjusting doctors fu to new ratio per clinic only
            results=np.empty((0,8))
            for i in new_appt_type_range:
                for j in fu_appt_type_range:
                    if 2*i+j<=8:#limits the total clinic time to 4 hours
                        params={**default_vals,'new_per_clinic_doc':i,'fu_per_clinic_doc':j}
                        result=run_sim(**params)
                        result=np.append(result,[i,j]).reshape(1,8)#add the new and fu type num
                        results=np.vstack((results,result))
            targ_wait=12# target wait time months
        #print('all results',results)
            filtered_results=results[results[:,2]<targ_wait]
            #print('filtered results',filtered_results)
            if filtered_results.size>0:
                min_idx=np.argmin(filtered_results[:,6])#filter by minimum new patinet slots needed
                print(min_idx)
                print(f'for your service with an annual referral rate of {filtered_results[min_idx,3]}, {filtered_results[min_idx,6]} new slots per clinic with  {filtered_results[min_idx,7]} fu slots would keep mean wait time less than {targ_wait} months')
            else:
                print('No work force combination in that range can meet your wait target. extend the range or increase the target difference')            
            #print(results)        
            plot_3d=True    
            if plot_3d:
                docs = results[:, 4]  
                nurses = results[:, 5]  
                min_time = results[:, 0]
                max_time = results[:, 1] 
                mean_time= results[:, 2]
                ref_rate=results[:,3]
                new_appts = results[:,6]
                fu_appts= results[:,7]
                #print(new_appts,fu_appts)
                # Create plot
                fig = plt.figure(figsize=(10, 8))
                ax = fig.add_subplot(projection='3d')  # 3D plot

            

                # Define grid for the plane
                x_range = np.linspace(np.min(new_appts), np.max(new_appts),50)  # Adjust based on your data range
                y_range = np.linspace(np.min(fu_appts), np.max(fu_appts), 50)
                #print(x_range,y_range)
                X, Y = np.meshgrid(x_range, y_range)

                Z = np.full_like(X, 12)  # Creates a flat plane at z = 12
                Z1= griddata((new_appts,fu_appts),mean_time,(X,Y),method='linear')
                Z2= griddata((new_appts,fu_appts),min_time,(X,Y),method='linear')
                Z3= griddata((new_appts,fu_appts),max_time,(X,Y),method='linear')
                #print(Z,Z1,Z2,Z3)
                
                    # Create the plotly surface objects
                fig = go.Figure()

                fig.add_trace(go.Surface(x=X, y=Y, z=Z2, colorscale=[[0,'Red'],[1,'Red']], opacity=0.1, name='Lower Bound',showscale=False))
                fig.add_trace(go.Surface(x=X, y=Y, z=Z1, colorscale='Blues_r', opacity=1, name='Mean Time',showscale=True,colorbar=dict(title="Wait Time (months)")))
                fig.add_trace(go.Surface(x=X, y=Y, z=Z3, colorscale=[[0,'Red'],[1,'Red']], opacity=0.1, name='Upper Bound',showscale=False))
                fig.add_trace(go.Surface(x=X, y=Y, z=Z, colorscale=[[0,'Gray'],[1,'Gray']], opacity=0.4, name='12 month boundary',showscale=False))

                # Update layout
                fig.update_layout(
                    title=f'Interpolated Wait Times for different new and f/u compositions for doctors with referral rate {ref_rate[0]}',
                    scene=dict(
                        xaxis_title='New appt per clinic',
                        yaxis_title='Follow up appts per clinic',
                        zaxis_title='Wait Time (months)'
                    ),
                    annotations=[
                    dict(x=0, y=1.05 ,text=f'{filtered_results[min_idx,6]} new slots per clinic with  {filtered_results[min_idx,7]} fu slots would keep mean wait time less than {targ_wait} months',showarrow=False, arrowhead=2)
                    ],
                    #width=900,
                    #height=900
                )
                
                st.plotly_chart(fig, use_container_width=True)
                #fig.show()
                #fig.write_html("service_model.html")    

        elif explore_fu_rate:
            '''Explore the variation in follow up rates for each condition'''
            results_ADHD=np.empty((0,8))
            results_ASD=np.empty((0,8))
            results_Complex=np.empty((0,8))
            for i in fu_range:
                for j in reg_fu_range:
                    params={**default_vals,'ADHD_fu_num':i,'ADHD_reg_fu_num':j}
                    result=run_sim(**params)
                    result=np.append(result,[i,j]).reshape(1,8)#add the fu nums
                    results_ADHD=np.vstack((results_ADHD,result))
                    params={**default_vals,'ASD_fu_num':i,'ASD_reg_fu_num':j}
                    result=run_sim(**params)
                    result=np.append(result,[i,j]).reshape(1,8)#add the fu nums
                    results_ASD=np.vstack((results_ASD,result))
                    params={**default_vals,'Complex_fu_num':i,'Complex_reg_fu_num':j}
                    result=run_sim(**params)
                    result=np.append(result,[i,j]).reshape(1,8)#add the fu nums
                    results_Complex=np.vstack((results_Complex,result))               
            targ_wait=12# target wait time months
        #print('all results',results)
            #print('filtered results',filtered_results)
                    
            #print(results)        
            plot_3d=True    
            if plot_3d:
                docs = results_ADHD[:, 4]  
                nurses = results_ADHD[:, 5]  
                mean_time_ADHD= results_ADHD[:, 2]
                mean_time_ASD= results_ASD[:, 2]
                mean_time_Complex= results_Complex[:, 2]
                ref_rate=results_ADHD[:,3]
                diag_fu_appts = results_ADHD[:,6]#this range is teh same for all conditions
                reg_fu_appts= results_ADHD[:,7]
                #print(new_appts,fu_appts)
                # Create plot
                fig = plt.figure(figsize=(10, 8))
                ax = fig.add_subplot(projection='3d')  # 3D plot

            

                # Define grid for the plane
                x_range = np.linspace(np.min(diag_fu_appts), np.max(diag_fu_appts),50)  # Adjust based on your data range
                y_range = np.linspace(np.min(reg_fu_appts), np.max(reg_fu_appts), 50)
                #print(x_range,y_range)
                X, Y = np.meshgrid(x_range, y_range)

                Z = np.full_like(X, 12)  # Creates a flat plane at z = 12
                Z1= griddata((diag_fu_appts,reg_fu_appts),mean_time_ADHD,(X,Y),method='linear')
                Z2= griddata((diag_fu_appts,reg_fu_appts),mean_time_ASD,(X,Y),method='linear')
                Z3= griddata((diag_fu_appts,reg_fu_appts),mean_time_Complex,(X,Y),method='linear')
                #print(Z,Z1,Z2,Z3)
                
                    # Create the plotly surface objects
                fig = go.Figure()

                fig.add_trace(go.Surface(x=X, y=Y, z=Z2, colorscale='Greens_r', opacity=0.5, name='Mean for ASD',showscale=True,colorbar=dict(title="ASD",x=0.8)))
                fig.add_trace(go.Surface(x=X, y=Y, z=Z1, colorscale='Blues_r', opacity=0.5, name='Mean for ADHD',showscale=True,colorbar=dict(title="ADHD",x=0.85)))
                fig.add_trace(go.Surface(x=X, y=Y, z=Z3, colorscale='Reds_r', opacity=0.5, name='Mean for Complex',showscale=True,colorbar=dict(title="Complex",x=0.9)))
                fig.add_trace(go.Surface(x=X, y=Y, z=Z, colorscale=[[0,'Gray'],[1,'Gray']], opacity=0.2, name='12 month boundary',showscale=False))

                # Update layout
                fig.update_layout(
                    title=f'Interpolated Wait Times for different f/u rates for different conditions with referral rate {ref_rate[0]}',
                    scene=dict(
                        xaxis_title='Diagnostic Follow up appt per patient',
                        yaxis_title='Regular Follow up appts per patient',
                        zaxis_title='Wait Time (months)'
                    ),showlegend=True,
                    #width=900,
                    #height=900
                )
                st.plotly_chart(fig, use_container_width=True)
                #fig.show()
                #fig.write_html("fu_rates_model.html")    


        else:
            results=np.empty((0,6))
            result=run_sim(**default_vals)
            results=np.vstack((results,result))
            print(f'for your service with an annual referral rate of {results[0,3]} the mean wait time is {results[0,2]} months with upper bound {results[0,1]} and lower bound {results[0,0]} months ')
        
            st.write(f'for your service with an annual referral rate of {results[0,3]} the mean wait time is {results[0,2]} months with upper bound {results[0,1]} and lower bound {results[0,0]} months ')
        #print(results.shape)
        #print(results)


        #run streamlit using python -m streamlit run model_code.py
