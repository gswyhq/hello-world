def Excel2Html(filename):
    import os
    from xlrd import open_workbook,formatting

    ext = filename.decode('utf-8')[-4:].encode('utf-8')
    if ext == '.pdf':
        return ''

    filepath = os.path.join(settings.MEDIA_ROOT, filename)

    wb = open_workbook(filepath,formatting_info=True)
    sheet = wb.sheet_by_index(0)
    html = '<table class="previewtable" border="1" cellpadding="1" cellspacing="1">'

    mergedcells={}
    mergedsapn={}
    mergedcellvalue={}
    for crange in sheet.merged_cells:
        rlo, rhi, clo, chi = crange
        for rowx in xrange(rlo, rhi):
            for colx in xrange(clo, chi):
                mergedcells[(rowx,colx)]=False
                value = str(sheet.cell_value(rowx,colx))
                if value.strip() != '':
                    mergedcellvalue[(rlo,clo)]=value

        mergedcells[(rlo,clo)]=True
        mergedsapn[(rlo,clo)]=(rhi-rlo, chi-clo)
        mergedsapn[(rlo,clo)]=(rhi-rlo, chi-clo)


    for row in xrange(sheet.nrows):
        html=html+'<tr>'
        for col in xrange(sheet.ncols):
            if (row,col) in mergedcells:
                if mergedcells[(row,col)]==True:
                    rspan,cspan = mergedsapn[(row,col)]
                    value = ''
                    if (row,col) in mergedcellvalue:
                        value = mergedcellvalue[(row,col)]
                    html=html+'<td rowspan=%s colspan=%s contenteditable="true">%s</td>'  % (rspan, cspan, value)
            else:
                value =sheet.cell_value(row,col)
                html=html+'<td contenteditable="true">' + str(value) + '</td>'

        html=html+'</tr>'

    html=html+'</table>'

    return html

