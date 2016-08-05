for entry in "$1"/*
do
	file=$(jpeginfo "$entry" | grep ERROR | cut -f 1 -d " ")
	if [ -n "$file" ]; then
		rm -f $file
	fi
done
