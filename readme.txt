export RAFT_PATH=raft
conda activate cuml_dev
mvn -DskipTests -Drat.skip=true -Dmaven.javadoc.skip=true -Dskip -Dmaven.scalastyle.skip=true  clean install
