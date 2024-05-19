for mode in "heatx" "heaty"
do
    for space in "original" "mp"
    do 
        python3 plot.py --override e2e_all_citation_node.yaml plot_mode $mode plot_space $space
        python3 plot.py --override e2e_all_commerce_node.yaml plot_mode $mode plot_space $space 
    done 
done