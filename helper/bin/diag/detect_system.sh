# detect_system.sh

if [ -n "$BASH_VERSION" ]; then
    echo "✅ Running in bash (version $BASH_VERSION)"
else
    echo "❌ Not running in bash"
fi

(return 0 2>/dev/null) || { echo "❌ Please source this script instead of executing it."; exit 1; }
echo "✅ It seems that this script is being included (rather than beeing called)"

HOSTNAME=$(hostname)  # this does not work well for lumi
if [[ $SYSTEM == "" ]]; then
   if [[ "$HOSTNAME" == roihu* ]]; then
      export SYSTEM=roihu
   elif [[ "$HOSTNAME" == mahti* ]]; then
      export SYSTEM=mahti
   elif [[ "$HOSTNAME" == puhti* ]]; then
      export SYSTEM=puhti
   elif grep -q 'lumi-super' /etc/motd; then
      export SYSTEM=lumi
   else
      echo "⚠️ Unknown system: $HOSTNAME. Please edit this script (detect_system.sh) manually."
      exit 1
   fi
fi
echo "✅ This seems to be \$HOSTNAME=$HOSTNAME node of the \$SYSTEM=$SYSTEM system"
