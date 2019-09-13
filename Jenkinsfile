#!/usr/bin/env groovy
// This shared library is available at https://github.com/ROCmSoftwarePlatform/rocJENKINS/
@Library('rocJenkins') _

// This is file for internal AMD use.
// If you are interested in running your own Jenkins, please raise a github issue for assistance.

import com.amd.project.*
import com.amd.docker.*

////////////////////////////////////////////////////////////////////////
// Mostly generated from snippet generator 'properties; set job properties'
// Time-based triggers added to execute nightly tests, eg '30 2 * * *' means 2:30 AM
properties([
    pipelineTriggers([cron('0 1 * * *'), [$class: 'PeriodicFolderTrigger', interval: '5m']]),
    buildDiscarder(logRotator(
      artifactDaysToKeepStr: '',
      artifactNumToKeepStr: '',
      daysToKeepStr: '',
      numToKeepStr: '10')),
    disableConcurrentBuilds(),
    [$class: 'CopyArtifactPermissionProperty', projectNames: '*']
   ])

import java.nio.file.Path;

rocSOLVERCI:
{

    def rocsolver = new rocProject('rocSOLVER')

    // Define test architectures, optional rocm version argument is available
    def nodes = new dockerNodes(['gfx900 && ubuntu', 'gfx906 && ubuntu'], rocsolver)

    boolean formatCheck = false
    rocsolver.timeout.compile = 180

    def compileCommand =
    {
        platform, project->

        project.paths.construct_build_prefix()

        def command = """#!/usr/bin/env bash
                    set -x
                    cd ${project.paths.project_build_prefix}
                    sudo mkdir build && cd build
                    export PATH=/opt/rocm/bin:$PATH
                    sudo cmake -DCMAKE_CXX_COMPILER=/opt/rocm/bin/hcc ..
                    sudo make 
                    """

        platform.runCommand(this, command)
    }

    def testCommand =
    {
        platform, project->

        def command

        if(auxiliary.isJobStartedByTimer())
        {
            command = """#!/usr/bin/env bash
                    set -x
                    cd ${project.paths.project_build_prefix}/build/clients/staging
                    LD_LIBRARY_PATH=/opt/rocm/hcc/lib GTEST_LISTENER=NO_PASS_LINE_IN_LOG sudo ./rocsolver-test --gtest_output=xml --gtest_color=yes --gtest_filter=*daily_lapack*
                """

            platform.runCommand(this, command)
            junit "${project.paths.project_build_prefix}/build/clients/staging/*.xml"
        }
        else
        {
            command = """#!/usr/bin/env bash
                    set -x
                    cd ${project.paths.project_build_prefix}/build/clients/staging
                    LD_LIBRARY_PATH=/opt/rocm/hcc/lib GTEST_LISTENER=NO_PASS_LINE_IN_LOG sudo ./rocsolver-test --gtest_output=xml --gtest_color=yes  --gtest_filter=*checkin_lapack*
                """
            
            platform.runCommand(this, command)
            junit "${project.paths.project_build_prefix}/build/clients/staging/*.xml"
        }
    }

    def packageCommand =
    {
        platform, project->

        def command

        if(platform.jenkinsLabel.contains('hip-clang'))
        {
            packageCommand = null
        }
        else if(platform.jenkinsLabel.contains('centos'))
        {
            command = """
                    set -x
                    cd ${project.paths.project_build_prefix}/build
                    sudo make package
                    sudo mkdir -p package
                    sudo mv *.rpm package/
                    sudo rpm -qlp package/*.rpm
                """

            platform.runCommand(this, command)
            platform.archiveArtifacts(this, """${project.paths.project_build_prefix}/build/package/*.rpm""")
        }
        else
        {
            command = """
                    set -x
                    cd ${project.paths.project_build_prefix}/build
                    sudo make package
                    sudo mkdir -p package
                    sudo mv *.deb package/
                """

            platform.runCommand(this, command)
            platform.archiveArtifacts(this, """${project.paths.project_build_prefix}/build/package/*.deb""")
        }
    }

    buildProject(rocsolver, formatCheck, nodes.dockerArray, compileCommand, testCommand, packageCommand)

}
