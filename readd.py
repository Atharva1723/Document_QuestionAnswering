import tabula

# Read the PDF file and extract the table data
table_data = tabula.read_pdf('Travel Policy.pdf', pages='all')

# Print the table data
print(table_data)