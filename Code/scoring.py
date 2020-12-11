def score_claim ():
    
    # Converting to json 
    data = request.get_json()
    user = data['user']
    password = data['password']
    policy_num = data['policy_num']
    
    # Finding the policy in database 
    claim = db.policy_pull(policy_num, user, password)

    # Applying capping values... 

    # Removing columns not in training
    claim = claim[originalColumns]

    # Grabbing only the numeric columns
    numericCols = pd.DataFrame(claim.select_dtypes(exclude=['object'])).columns
    
    for var in numericCols:  
    
        feature = claim[var]

        # Calculate boundaries
        lower = capping.loc[capping.feature == var].lower.values[0]
        upper = capping.loc[capping.feature == var].upper.values[0]
        # Replace outliers
        claim[var] = np.where(feature > upper, upper, np.where(feature < lower, lower, feature))

    return 'success'