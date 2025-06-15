import pandas as pd
#import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
#import sklearn
#from sklearn.model_selection import train_test_split
#from sklearn.regressor import LinearRegression


# Read the CSV file
data = pd.read_csv('main.csv')


# Splitting the 'Country, Relations' column
relations_split = data['Country, Relations'].str.split(',', n=1, expand=True)
data['Country_relations'] = relations_split[0]
data['Relations'] = relations_split[1]

# Dropping the original 'Country, Relations' column
data.drop(['Country, Relations'], axis=1, inplace=True)

# Renaming the country columns to be consistent
data.rename(columns={'Country.1': 'Country_military', 'Country.2': 'Country_democracy', 'Military ': 'Military'}, inplace=True)

# Separate the data into different DataFrames
gini_data = data[['Country', 'Gini']]
gdp_data = data[['Country', 'GDP']]
military_data = data[['Country_military', 'Military']].rename(columns={'Country_military': 'Country'})
democracy_data = data[['Country_democracy', 'Democracy']].rename(columns={'Country_democracy': 'Country'})

# Plotting the Gini coefficients
plt.figure(figsize=(12, 8))
sns.set_style('whitegrid')
sns.set_context('paper')
plt.title('Gini Coefficients')
plt.xlabel('Countries')
plt.ylabel('Gini Coefficient')
plt.xticks(rotation=90)
plt.plot(gini_data['Country'].astype(str), gini_data['Gini'], marker='o', markersize=10, linestyle='--', color='#1f77b4')

plt.savefig('gini_coefficients.png')
plt.close()
plt.clf()
plt.cla()
plt.close()


# Merging the DataFrames based on the country name
data_merged = pd.merge(gini_data, gdp_data, on='Country', how='left')
data_merged = pd.merge(data_merged, military_data, on='Country', how='left')
data_merged = pd.merge(data_merged, data[['Country_relations', 'Relations']].rename(columns={'Country_relations': 'Country'}), on='Country', how='left')
data_merged = pd.merge(data_merged, democracy_data, on='Country', how='left')
data_merged.drop(['Country_democracy', 'Country_relations', 'Relations'], axis=1, errors='ignore', inplace=True)
data_merged.rename(columns={'Country_military': 'Country'}, inplace=True)
data_merged.dropna(inplace=True)
data_merged.drop_duplicates(inplace=True)
data_merged.reset_index(drop=True, inplace=True)
data_merged.drop(['Gini', 'GDP', 'Military'], axis=1, inplace=True)
data_merged.to_csv('main_merged.csv', index=False)
data_merged.head()
data_merged.shape
data_merged.describe()
data_merged.info()
data_merged.isnull().sum()
data_merged.isnull().sum() / len(data_merged) * 100
# Stop execution before additional exploratory code
import sys
sys.exit()

# Define a mapping from incorrect country names to correct ones
country_name_mapping = country_name_mapping = {'Albania' :'Albania','Algeria' :'Algeria''Angola' 'Angola', 'Argentina' : 'Argentina', 'Armenia' : 'Armenia', 'Australia' : 'Australia', 'Austria' : 'Austria', 'Azerbaijan' : 'Azerbaijan', 'Bahamas' : 'Bahamas', 'Bahrain' : 'Bahrain', 'Bangladesh' : 'Bangladesh', 'Barbados' : 'Barbados', 'Belarus' : 'Belarus', 'Belgium' : 'Belgium', 'Belize' : 'Belize', 'Benin' : 'Benin', 'Bhutan' : 'Bhutan', 'Bolivia' : 'Bolivia', 'Bosnia and Herzegovina' : 'Bosnia and Herzegovina', 'Botswana' : 'Botswana', 'Brazil' : 'Brazil', 'Brunei' : 'Brunei', 'Bulgaria' : 'Bulgaria', 'Burkina Faso' : 'Burkina Faso', 'Burundi' : 'Burundi', 'Cambodia' : 'Cambodia', 'Cameroon' : 'Cameroon', 'Canada' : 'Canada', 'Central African Republic' : 'Central African Republic', 'Chad' : 'Chad', 'Chile' : 'Chile', 'China' : 'China', 'Colombia' : 'Colombia', 'Comoros' : 'Comoros', 'Congo' : 'Congo', 'Costa Rica' : 'Costa Rica', 'Croatia' : 'Croatia', 'Cuba' : 'Cuba', 'Cyprus' : 'Cyprus', 'Czech Republic' : 'Czech Republic', 'Denmark' : 'Denmark', 'Djibouti' : 'Djibouti', 'Dominica' : 'Dominica', 'Dominican Republic' : 'Dominican Republic', 'Ecuador' : 'Ecuador', 'Egypt' : 'Egypt', 'El Salvador' : 'El Salvador', 'Equatorial Guinea' : 'Equatorial Guinea', 'Eritrea' : 'Eritrea', 'Estonia' : 'Estonia', 'Ethiopia' : 'Ethiopia', 'Fiji' : 'Fiji', 'Finland' : 'Finland', 'France' : 'France', 'Gabon' : 'Gabon', 'Gambia' : 'Gambia', 'Georgia' : 'Georgia', 'Germany' : 'Germany', 'Ghana' : 'Ghana', 'Greece' : 'Greece', 'Grenada' : 'Grenada', 'Guatemala' : 'Guatemala', 'Guinea' : 'Guinea', 'Guinea-Bissau' : 'Guinea-Bissau', 'Guyana' : 'Guyana', 'Haiti' : 'Haiti', 'Honduras' : 'Honduras', 'Hungary' : 'Hungary', 'Iceland' : 'Iceland', 'India' : 'India', 'Indonesia' : 'Indonesia', 'Iran' : 'Iran', 'Iraq' : 'Iraq', 'Ireland' : 'Ireland', 'Israel' : 'Israel', 'Italy' : 'Italy', 'Jamaica' : 'Jamaica', 'Japan' : 'Japan', 'Jordan' : 'Jordan', 'Kazakhstan' : 'Kazakhstan', 'Kenya' : 'Kenya', 'Kiribati' : 'Kiribati', 'Kosovo' : 'Kosovo', 'Kuwait' : 'Kuwait', 'Kyrgyzstan' : 'Kyrgyzstan', 'Laos' : 'Laos', 'Latvia' : 'Latvia', 'Lebanon' : 'Lebanon', 'Lesotho' : 'Lesotho', 'Liberia' : 'Liberia', 'Libya' : 'Libya', 'Liechtenstein' : 'Liechtenstein', 'Lithuania' : 'Lithuania', 'Luxembourg' : 'Luxembourg', 'Madagascar' : 'Madagascar', 'Malawi' : 'Malawi', 'Malaysia' : 'Malaysia', 'Maldives' : 'Maldives', 'Mali' : 'Mali', 'Malta' : 'Malta', 'Marshall Islands' : 'Marshall Islands', 'Mauritania' : 'Mauritania', 'Mauritius' : 'Mauritius', 'Mexico' : 'Mexico', 'Micronesia' : 'Micronesia', 'Moldova' : 'Moldova', 'Monaco' : 'Monaco', 'Mongolia' : 'Mongolia', 'Montenegro' : 'Montenegro', 'Morocco' : 'Morocco', 'Mozambique' : 'Mozambique', 'Myanmar' : 'Myanmar', 'Namibia' : 'Namibia', 'Nauru' : 'Nauru', 'Nepal' : 'Nepal', 'Netherlands' : 'Netherlands', 'New Zealand' : 'New Zealand', 'Nicaragua' : 'Nicaragua', 'Niger' : 'Niger', 'Nigeria' : 'Nigeria', 'Norway' : 'Norway', 'Oman' : 'Oman', 'Pakistan' : 'Pakistan', 'Palau' : 'Palau', 'Panama' : 'Panama', 'Papua New Guinea' : 'Papua New Guinea', 'Paraguay' : 'Paraguay', 'Peru' : 'Peru', 'Philippines' : 'Philippines', 'Poland' : 'Poland', 'Portugal' : 'Portugal', 'Qatar' : 'Qatar', 'Romania' : 'Romania', 'Russia' : 'Russia', 'Rwanda' : 'Rwanda', 'Saint Kitts and Nevis' : 'Saint Kitts and Nevis', 'Saint Lucia' : 'Saint Lucia',
'Saint Vincent and the Grenadines' : 'Saint Vincent and the Grenadines', 'Samoa' : 'Samoa', 'San Marino' : 'San Marino', 'Sao Tome and Principe' : 'Sao Tome and Principe', 'Saudi Arabia' : 'Saudi Arabia', 'Senegal' : 'Senegal', 'Serbia' : 'Serbia', 'Seychelles' : 'Seychelles', 'Sierra Leone' : 'Sierra Leone', 'Singapore' : 'Singapore', 'Slovakia' : 'Slovakia', 'Slovenia' : 'Slovenia', 'Solomon Islands' : 'Solomon Islands', 'Somalia' : 'Somalia', 'South Africa' : 'South Africa', 'South Korea' : 'South Korea', 'Spain' : 'Spain', 'Sri Lanka' : 'Sri Lanka', 'Sudan' : 'Sudan', 'Suriname' : 'Suriname', 'Sweden' : 'Sweden', 'Switzerland' : 'Switzerland', 'Syria' : 'Syria', 'Taiwan' : 'Taiwan', 'Tajikistan' : 'Tajikistan', 'Tanzania' : 'Tanzania', 'Thailand' : 'Thailand', 'Togo' : 'Togo', 'Tonga' : 'Tonga', 'Trinidad and Tobago' : 'Trinidad and Tobago', 'Tunisia' : 'Tunisia', 'Turkey' : 'Turkey', 'Turkmenistan' : 'Turkmenistan', 'Tuvalu' : 'Tuvalu', 'Uganda' : 'Uganda', 'Ukraine' : 'Ukraine', 'United Arab Emirates' : 'United Arab Emirates', 'United Kingdom' : 'United Kingdom', 'United States' : 'United States', 'Uruguay' : 'Uruguay', 'Uzbekistan' : 'Uzbekistan', 'Vanuatu' : 'Vanuatu', 'Vatican City' : 'Vatican City', 'Venezuela' : 'Venezuela', 'Vietnam' : 'Vietnam', 'Yemen' : 'Yemen', 'Zambia' : 'Zambia', 'Zimbabwe' : 'Zimbabwe'}
gini_data = data[['Country', 'Gini']].copy()
gini_data['Country'] = gini_data['Country'].replace(country_name_mapping)

# Apply the mapping to correct the country names
gdp_data = data[['Country', 'GDP']].copy() 
gdp_data['Country'] = gdp_data['Country'].replace(country_name_mapping)
gdp_data['Country'].replace(country_name_mapping)
military_data = data[['Country_military', 'Military']].copy()
military_data['Country_military'] = military_data['Country_military'].replace(country_name_mapping)
democracy_data['Country'] = democracy_data['Country'].replace(country_name_mapping)


print(set(gini_data['Country']) - set(gdp_data['Country']))
print(set(military_data['Country_military']) - set(gdp_data['Country']))
print(set(democracy_data['Country']) - set(gdp_data['Country']))

# Plotting the Gini coefficients
plt.figure(figsize=(12, 8))
sns.set_style('whitegrid')
sns.set_context('paper')
plt.title('Gini Coefficients')
plt.xlabel('Countries')
plt.ylabel('Gini Coefficient')
plt.xticks(rotation=90)
plt.plot(gini_data['Country'], gini_data['Gini'], marker='o',
         markersize=10, linestyle='--', color='#1f77b4')
plt.show()
plt.savefig('gini_coefficients.png')
plt.close()
plt.clf()
plt.cla()
plt.close()

# Merging the DataFrames based on the country name
data_merged = pd.merge(gini_data, gdp_data, on='Country', how='left')



# Now you have a merged DataFrame with all the information
print(data_merged.head())
