import matplotlib.pyplot as plt
import matplotlib as mpl
import os
import time

def visualize_overlaps(plag_spans, thresh_to_detected_spans, **metadata):
    '''
    <plag_spans> is a list of plagiarized spans
    <thresh_to_detected_spans> is a dict of form thresh -> [list of detected spans]
    '''
    plt.new()
    bottom = 0
    top_row = len(thresh_to_detected_spans)
    row_height = 1.0 / top_row
    sorted_thresh = sorted(thresh_to_detected_spans.keys())

    for row_num, thresh in enumerate(sorted_thresh):
        detected_spans = thresh_to_detected_spans[thresh]
        for dspan_start, dspan_end in detected_spans:
            width = dspan_end - dspan_start
            plt.barh(1 + row_num, width, height=row_height, align='center', left=dspan_start, color='blue', edgecolor='blue')

    for pspan_start, pspan_end in plag_spans:
        width = pspan_end - pspan_start
        print pspan_start, pspan_end
        print width
        plt.barh(bottom, width, height=row_height, align='center', left=pspan_start, color='red', edgecolor='red')

    ylabels = ['', 'Actual Plag'] + sorted_thresh + ['']
    ytick_nums = range(-1, top_row + 2)
    plt.yticks(ytick_nums, ylabels)

    file_name = ''
    if 'doc_name' in metadata:
        plt.title(metadata['doc_name'])
        file_name += metadata['doc_name']
    
    file_name += '_' + str(time.time()) + '.pdf'

    path = os.path.join(os.path.dirname(__file__), "../figures/overlap_viz/" + file_name)
    plt.savefig(path)
    plt.close()

def visualize_confidence_overaps(plag_spans, detection_methods, list_of_spans, list_of_plag_confidences, **metadata):
    '''
    <plag_spans> is a list of plagiarized spans
    <detection_methods> is a list of strings representing the different methods used 
    where detection_methods[i] corresponds to the spans stored in list_of_spans[i]
    and the confidences stored in list_of_plag_confidences[i]
    '''
    bottom = 0
    top_row = len(detection_methods)
    row_height = 1.0 / top_row

    cmap = mpl.colors.Colormap('binary')
    #cmap = mpl.colors.Colormap('RdBu')
    # TODO insert plag_spans in the middle of the confidences
    
    for row_num, dmethod in enumerate(detection_methods):
        detected_spans = list_of_plag_confidences[thresh]
        for dspan_start, dspan_end in detected_spans:
            width = dspan_end - dspan_start
            plt.barh(1 + row_num, width, height=row_height, align='center', left=dspan_start, color='blue', edgecolor='blue')

    for pspan_start, pspan_end in plag_spans:
        width = pspan_end - pspan_start
        print pspan_start, pspan_end
        print width
        plt.barh(bottom, width, height=row_height, align='center', left=pspan_start, color='red', edgecolor='red')

    ylabels = ['', 'Actual Plag'] + sorted_thresh + ['']
    ytick_nums = range(-1, top_row + 2)
    plt.yticks(ytick_nums, ylabels)

    file_name = ''
    if 'doc_name' in metadata:
        plt.title(metadata['doc_name'])
        file_name += metadata['doc_name']
    
    file_name += '_' + str(time.time()) + '.pdf'

    path = os.path.join(os.path.dirname(__file__), "../figures/overlap_viz/" + file_name)
    plt.savefig(path)
    plt.close()



if __name__ == '__main__':
    pass