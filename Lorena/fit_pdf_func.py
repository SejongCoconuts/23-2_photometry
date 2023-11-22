############# PDF Calculation ############# 

def pdf_cal(FLUX):
    data = FLUX
 
    # Fit a log-normal distribution to the data and estimate parameters
    lnsig, loc, scale = lognorm.fit(data, floc=0)
    
    # Calculate the mean (lnmu)
    lnmu = loc - (lnsig**2) / 2

    # Generate PDF
    data_pdf2 = np.linspace(0, data.max(), 1000)
    pdf_light2 = lognorm.pdf(data_pdf2, lnsig, loc, scale)
    
    fig, g = plt.subplots(figsize=(10, 5))
    fig.set_facecolor('white')
    g.plot(data_pdf2, pdf_light2, 'r-', lw=2, label='log-normal distribution PDF')
    g.hist(data, alpha = 0.5, bins=30, density=True, color = 'b', label=r'Photon Flux', edgecolor = 'black')
    g.set_ylabel('Probability Density Fuction',fontsize=15)
    g.legend(loc ='best')
    for spine in ['top', 'right','bottom','left']:
        g.spines[spine].set_linewidth(2)
    g.tick_params(labelsize=10,length=3,width=2)
    return lnmu, lnsig, data_pdf2, pdf_light2
