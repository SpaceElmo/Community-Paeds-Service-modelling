import numpy as np
import matplotlib.pyplot as plt
import csv
import random as rand
import math
import scipy.stats as stats
from mpl_toolkits.mplot3d import Axes3D


'''This code is a model of all non statory work to determine wait times dependant on referral numbers and age distributions. Capacity is calculated annually as it scales linearly with time similarly to demand. '''

##Global default Variables. diagnosis rates for each age group <6 and >6 were calculated form the 24/25 referral sheet. This covered 88% of referrals with the remainder being referrrals##
Total_ref_num=700
num_of_years=1
num_docs=4 #WTE doctors
num_nurses=3 #WTE nurses
new_per_clinic_doc=1 #no new patient appt
fu_per_clinic_doc=4 #no of f/u appt
new_per_clinic_nurse=0 #no of new patient appt
fu_per_clinic_nurse=4 #no of f/u appt
num_of_clinics_docs = 4#per week per doc
num_of_clinics_nurses = 3#per week per nurse
ADHD_fu_num=2#number of follow ups always needed post new assessment
ADHD_reg_fu_num=2#number of folloow ups needed per year normally for each patient
ASD_fu_num=3#number of follow ups always needed post new assessment
ASD_reg_fu_num=0.2#number of follow ups needed per year normally
Complex_fu_num=4#number of follow ups always needed post new assessment
Complex_reg_fu_num=2#number of follow ups needed per year normally
min_age=1.5
max_age=17

free_var_list=[Total_ref_num,num_of_years,
num_docs,
num_nurses,
new_per_clinic_doc, #length of time of new patient appt
fu_per_clinic_doc, #length of time of f/u appt
new_per_clinic_nurse,
fu_per_clinic_nurse,
num_of_clinics_docs,
num_of_clinics_nurses,
ADHD_fu_num,
ADHD_reg_fu_num,
ASD_fu_num,
ASD_reg_fu_num,
Complex_fu_num,
Complex_reg_fu_num,
min_age,
max_age]

#Hard coded variables in functions(cannot alter them here)
'''Working_weeks=42
ASD_frac_under6=0.74
ADHD_frac_under6= 0.04
Complex_frac_under6=0.22
ASD_frac_over6=0.27
ADHD_frac_over6= 0.43
Complex_frac_over6=0.3'''

#functions
#model the age distribution. This model is based on observations of the actual distribution of age at referral over the last 3 years using a skewed normal fit
def find_age_dist(Total_ref_num,min_age,max_age,delay=1,plot=False):
    '''delay here refers to teh fact that teh distrubution of age when seen will be delayed by the wait time and is hardcoded. This is modelled simply as adding to the peak. default is 1'''
    age_data= np.genfromtxt('ages.csv',delimiter=',',skip_header=1)
    skew,peak_age,peak_sd=stats.skewnorm.fit(age_data)
    print(f'fitted stats are skew {skew}, peak age {peak_age}, sd {peak_sd}')
    age_dist=stats.skewnorm.rvs(a=skew, loc=peak_age+delay, scale=peak_sd ,size=Total_ref_num)
    age_dist=np.clip(age_dist,min_age,max_age)
    if plot:
        plt.hist(age_dist, bins=18, density=False, alpha=0.6, color='blue')
        plt.xlabel("Years")
        plt.ylabel("Density")
        plt.title("Skewed Normal Distribution")
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

def get_annual_fu_burdens(age_dist,total_new,total_fu):
    'returns the mean annual fu burden per new patient and the sd of that value. Also returns the mean and upper new to f/u ratios given the age and diagnosis distributions'
    lifetime_ratio=total_fu/total_new# Over the lifetime of the new patient what is the number of follows for a population. This does not vary significantly with ref rate but is dependant on underlying diagnosis and age distribution
    mean_annual_fu_burden=np.mean(lifetime_ratio/(18-age_dist))
    sd_annual_fu_burden=np.std(lifetime_ratio/(18-age_dist))
    upper_lim=np.ceil(mean_annual_fu_burden+sd_annual_fu_burden)
    lower_lim=np.ceil(mean_annual_fu_burden)
    return(mean_annual_fu_burden,sd_annual_fu_burden,upper_lim,lower_lim)
#print(f"the ideal ratio of new to follow up is 1:{int(upper_lim)} and minimum safe is 1:{int(lower_lim)}")

def get_annual_new_burden(Total_ref_num,num_of_years):
    new_annual_burden= Total_ref_num/num_of_years
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
def get_fu_demand(mean_annual_fu_burden,sd_annual_fu_burden,annual_new_burden):
    '''gets the annual follow up burden from the calculated ratio of new to f/u'''
    fu_demand_min=(mean_annual_fu_burden-sd_annual_fu_burden)*annual_new_burden#using the sd of fu burden as a proxy as it takes into account diagnosis and age
    fu_demand_max=(mean_annual_fu_burden+sd_annual_fu_burden)*annual_new_burden
    return(fu_demand_max,fu_demand_min)

def get_wait_times(fu_demand_min,fu_demand_max,new_demand,fu_capacity,new_capacity):
    min_wait_time=(fu_demand_min/fu_capacity)+(new_demand/new_capacity)
    max_wait_time=(fu_demand_max/fu_capacity)+(new_demand/new_capacity)
    #print(f'The lower bound wait_time using a service workforce of {num_docs} docs and {num_nurses} nurses is {min_wait_time*12} months. The upper bound wait time is {max_wait_time*12} months')
    return(min_wait_time,max_wait_time)


#########execute###########################################################################################################

#run the code with default vals but vary a line depending on range
age_dist=find_age_dist(Total_ref_num,min_age,max_age)
total_fu=get_ADHD_fu_burden(age_dist,ADHD_fu_num,ADHD_reg_fu_num)+get_ASD_fu_burden(age_dist,ASD_fu_num,ASD_reg_fu_num)+get_Complex_fu_burden(age_dist,Complex_fu_num,Complex_reg_fu_num)
mean_annual_fu,sd_annual_fu,upper_lim,lower_lim=get_annual_fu_burdens(age_dist,Total_ref_num,total_fu)
print(f"the ideal ratio of new to follow up is 1:{int(upper_lim)} and minimum safe is 1:{int(lower_lim)}")
annual_new=get_annual_new_burden(Total_ref_num,num_of_years)
num_docs_range=[2,3,4,5,6,7,8]
num_nurses_range=[1,2,3,4,5,6]
results=np.empty((0,4))
for i in num_docs_range:
    for j in num_nurses_range:
        fu_capacity=fu_capacity_docs(i,num_of_clinics_docs,fu_per_clinic_doc)+fu_capacity_nurses(j,num_of_clinics_nurses,fu_per_clinic_nurse)
        new_capacity=new_capacity_docs(i,num_of_clinics_docs,new_per_clinic_doc)+new_capacity_nurses(j,num_of_clinics_nurses,new_per_clinic_nurse)
        fu_demand_max,fu_demand_min=get_fu_demand(mean_annual_fu,sd_annual_fu,annual_new)
        min_wait_time,max_wait_time=get_wait_times(fu_demand_min,fu_demand_max,annual_new,fu_capacity,new_capacity)
        results=np.vstack((results,[i,j,min_wait_time*12,max_wait_time*12]))
        results[results<0]=0
print(results.shape)
print(results)   
        
plot_3d=False
plot_2d=True
# Extract columns
docs = results[:, 0]  # First independent variable
nurses = results[:, 1]  # Second independent variable
min_time = results[:, 2] # Dependent variable (third column)
max_time = results[:, 3] # Dependent variable (fourth column)

if plot_3d:
    # Create plot
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(projection='3d')  # 3D plot

    # Plot first independent set against z1
    ax.scatter(docs, nurses, min_time, color='blue', alpha=0.5, label="Lower bound wait time")

    # Plot first independent set against z2
    ax.scatter(docs, nurses, max_time, color='red', alpha=0.5, label="Upper bound wait time")

    # Define grid for the plane
    x_range = np.linspace(np.min(num_docs_range), np.max(num_docs_range), 30)  # Adjust based on your data range
    y_range = np.linspace(np.min(num_nurses_range), np.max(num_nurses_range), 30)

    X, Y = np.meshgrid(x_range, y_range)
    Z = np.full_like(X, 12)  # Creates a flat plane at z = 12
    ax.plot_surface(X, Y, Z, alpha=0.5, color='gray')  # Semi-transparent gray plane
    # Labels and title
    ax.set_xlabel("Number of doctors WTE")
    ax.set_ylabel("Number of nurses WTE")
    ax.set_zlabel("Wait time in months")
    ax.set_title("3D Scatter Plot of Independent vs Dependent Variables")
    ax.legend()

    plt.show()


if plot_2d:
# Define unique groups from Column 2
    unique_categories = num_nurses_range

    # Create two subplots
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Plot 1: Column 1 vs Column 3, categorized by Column 2
    for category in unique_categories:
        mask = nurses == category
        axes[0].plot(docs[mask], min_time[mask], marker='o', linestyle='-', label=f"No of nurses is {category:.1f}")

    axes[0].set_xlabel("Number of doctors WTE")
    axes[0].set_ylabel("Wait time (months)")
    axes[0].set_title("Plot 1: Doctors vs Lower bound time (Grouped by number of nurses WTE)")
    axes[0].legend()

    # Plot 2: Column 1 vs Column 4, categorized by Column 2
    for category in unique_categories:
        mask = nurses == category
        axes[1].plot(docs[mask], max_time[mask], marker='o', linestyle='-', label=f"No of nurses is {category:.1f}")

    axes[1].set_xlabel("Number of doctors WTE")
    axes[1].set_ylabel("Wait time (months)")
    axes[1].set_title("Plot 1: Doctors vs Upper bound time (Grouped by number of nurses WTE)")
    axes[1].legend()

    plt.tight_layout()
    plt.show()


