#!/usr/bin/env python
# coding: utf-8

# In[77]:


class MacroVariable:
    def __init__(self,
                 my_path,
                 metadata_file,
                 scenario):
        """
        Constructor of parent class.
        Args:
            my_path: string, location of the data folder
            metadata_file: string, file with variables description
            scenario: string, name of scenario data file
        Returns:
        -
        """
        self.my_path = my_path
        self.metadata_file = metadata_file
        self.scenario = scenario

    def get_value(self, x):
        """
        Re-Implement this method in child class. Gets
        quarterly predictions for variables.
        Args:
            variable: string with variable name
        Returns:
            matrix: matrix with predicted variable values
        Author:   John Doe, 2019/12/06
        Reviewer: Jack Doe, 2019/12/06
        """
       
        
        import pandas as pd
        import numpy as np
        df = pd.read_csv(self.scenario)
        x=df[df['Variable'] == 'SP500']
        x=x['Value']
        xx=np.asmatrix(x)
        xxx=np.log(xx)
        VIX=18.7825-0.7757*xxx

        return VIX

    @staticmethod
    def get_description():
        """
        Prints variable description.
        Args:
            -
        Returns:
            String: string with variable description
        Author: John Doe, 2019/12/06
        Reviewer: Jack Doe, 2019/12/06
        """
        import pandas as pd
        df = pd.read_csv(self.metadata_file)
        print(df['Description'])
        return 'Macro Variable'

    def get_initial_values(self, x):  # method to build!!!
        """
        Loads the initial scenario values for given variables.
        Args:
            variables: string vector with variable names required from scenario
        Returns:
            scen_values: Matrix with scenario values of required variables
        Author: John Doe, 2019/12/06
        Reviewer: Jack Doe, 2019/12/06
        
        """
        
        import pandas as pd
        df = pd.read_csv(self.scenario)
        SP500=df[df['Variable'] == 'SP500']
        x=SP500['Value']
        scen_values=np.asmatrix(x)
        return scen_values

    @staticmethod
    def transform_values_to_log(x):
        """
        Transforms values to log values.
        Args:
            x: matrix with values of required variables
        Returns:
            log(x): matrix with log values of required variables
        Author: John Doe, 2019/12/06
        Reviewer: Jack Doe, 2019/12/06
        """
        
        logx=np.log(scen_values)

        
        return logx

    @staticmethod
    def get_return_type():
        """
        Gets the return type of the variable.
        Args:
            -
        Returns:
            String: string with variable return type
        Author: John Doe, 2019/12/06
        Reviewer: Jack Doe, 2019/12/06
        """
        type(scen_values)
        return None

#######################################################################################################################
############################################ Child class ##############################################################
# Code here:


class VIX(MacroVariable):
    def __init__(self,
                 my_path,
                 metadata_file,
                 scenario):  # method to build!!!
        """ Constructor of child class.
        Args:
            my_path: string, location of the data folder
            metadata_file: string, file with variables description
            scenario: string, name of scenario data file
        Returns:
        -
        """
        super().__init__(my_path, metadata_file, scenario)
        self.indep_variable = ("SP500")  # examples of explanatory variables, to change/build!!!

    @staticmethod
    def get_description():
        """
        Prints VIX variable description.
        Args:
            -
        Returns:
            String: string with variable description
        Author:   John Doe, 2019/12/06
        Reviewer: Jack Doe, 2019/12/06
 
        """
        import pandas as pd
        df = pd.read_csv(self.metadata_file)
        print(df['Description'][1])
        return 'United States VIX Index (end of period)'

    @staticmethod
    def get_return_type():
        """
        Gets the return type of the VIX variable.
        Args:
            -
        Returns:
          String: string with variable return type
        Author:   John Doe, 2019/12/06
        Reviewer: Jack Doe, 2019/12/06
        """
        return 'level'

    def get_coefficients(self, coefficients):  # method to build!!!
        """
        Fetches coefficients of the model for provided file/model name.
        Args:
            coefficients: string with coefficient file name
        Returns:
            coefficients_matrix: matrix with coefficient names and values
        Author:   John Doe, 2019/12/06
        Reviewer: Jack Doe, 2019/12/06"""
        coefficients_matrix=np.matrix([[Intercept,Beta],[18.7825,-0.7757]])
        
        
        return coefficients_matrix

    def get_value(self, variable):  # method to build!!!
        """
        Gets quarterly predictions for given variables.
        The final output is in the matrix format consisting of
        predicted variable values and time periods in scenario
        horizon, i.e. variable name as column name and periods
        as row names.
        Args:
          variable: string with variable name
        Returns:
          Matrix: matrix with predicted variable values
        Author:   John Doe, 2019/12/06
        Reviewer: Jack Doe, 2019/12/06
        """
        import numpy as np
        import pandas as pd
        df = pd.read_csv(self.scenario)
        x=df[df['Variable'] == 'SP500']
        x=x['Value']
        xx=np.asmatrix(x)
        xxx=np.log(xx)
        VIX=18.7825-0.7757*xxx
        bb=np.squeeze(np.asarray(VIX))
        b=np. array(['0','1Q', '2Q','3Q','4Q', '5Q','6Q','7Q', '8Q','9Q'])
        arr = np.stack((b, bb), axis=1)
        import pandas as pd
        df = pd.DataFrame(arr,columns=['TimeStamp', 'VIX'])
        print(df)
        return VIX


# In[78]:


if __name__ == '__main__':
    my_path = ''  # folder path
    metadata_file = 'Metadata.csv'
    scenario = 'EconomicScenario.csv'
    vix_model = VIX(my_path, metadata_file, scenario)
    vix_model.get_value('VIX')


# In[ ]:





# In[ ]:




