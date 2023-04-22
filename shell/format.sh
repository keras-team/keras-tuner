isort keras_tuner
black keras_tuner

for i in $(find keras_tuner -name '*.py') # or whatever other pattern...
do
  if ! grep -q Copyright $i
  then
    echo $i
    cat shell/copyright.txt $i >$i.new && mv $i.new $i
  fi
done

flake8 keras_tuner
