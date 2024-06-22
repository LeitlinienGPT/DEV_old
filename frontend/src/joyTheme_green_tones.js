const joyTheme = {
  colorSchemes: {
    light: {
      palette: {
        primary: {
          background: '#DCE7CD',
          box: '#FDFDF6',
          botton_tonal: '#DCE7CD' ,
          button_before: '#C4F1A1', 
          button_after: '#44692B', 
          contrastText: '#000000',
          border: '#A8AF9D',
          border_thin: '#73776D',
        },
        // dark mode
        secondary: {
          background: '#404A38', // A calming blue color (Primary Main)
          button: '#2D5016', // Lighter shade of blue (Primary Light)
          button_soft: '#404A38', // Lighter shade of blue (Primary Light)
          contrastText: '#000000', // White for contrast on primary color
          border: '#676D63',
          border_thin: '#8E9287',
        },
        error: {
          main: '#f44336', // Red color for errors
        },
        warning: {
          main: '#ffa726', // Orange color for warnings
        },
        info: {
          main: '#29b6f6', // Light blue color for informational messages
        },
        success: {
          main: '#66bb6a', // Green color for success messages
        },
        // Additional color settings can be added as needed
      },
      components: {
        JoyButton: {
          styleOverrides: {
            root: { // This will target all buttons
              '&:hover': {
                backgroundColor: '#44692B', // button hovering
              },
              '&:active': {
                backgroundColor: '#7ECE36', // button pressed
              },
            },
          },
          variants: [
            {
              props: { variant: 'theme_paolo' }, // Assuming you want these styles for 'soft' variant buttons
              style: {
                backgroundColor: '#C4F1A1', // button_normal
                '&:hover': {
                  backgroundColor: '#44692B', // button hovering
                },
                '&:active': {
                  backgroundColor: '#7ECE36', // button pressed
                },
              },
            },
          ],
        },
      },
      
    },
    // You can also define a 'dark' scheme if needed
  },
};

export default joyTheme;